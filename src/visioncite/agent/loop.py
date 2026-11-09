"""最小 Agent 循环：模型决策 → 调工具 → 看结果 → 再决策。

为什么不用 LangChain/LangGraph？
    这个循环只有几十行：把工具定义给模型、执行它要求的调用、把结果塞回对话、
    直到模型给出最终答案。自己写一遍之后，框架的抽象（State/Node/Edge）
    反而更容易理解；也避免把项目绑在某个框架版本上。

安全边界：
    - 最多 ``agent_max_steps`` 步，防止模型反复调工具烧 token；
    - 最终答案必须能落到工具返回的证据上，否则弃答；
    - 模型调用失败时回退到抽取式作答，而不是报错。
"""

from __future__ import annotations

import json

from visioncite.agent.models import AgentResult, AgentStep
from visioncite.answer.qa import answer_question
from visioncite.answer.verify import verify_answer
from visioncite.config import Settings
from visioncite.errors import LLMError
from visioncite.llm.base import ChatMessage, LLMProvider, Usage
from visioncite.llm.factory import create_provider
from visioncite.logging import get_logger
from visioncite.pipeline import Document
from visioncite.tools.base import ToolRegistry
from visioncite.tools.document_tools import build_document_tools
from visioncite.types import Citation

logger = get_logger("agent")

SYSTEM_PROMPT = """你是 Document Agent，负责回答关于当前文档的问题。规则：
1. 先调用工具检索证据，再作答；不要凭记忆回答；
2. 答案必须来自工具返回的文本，不确定就直说"文档中没有找到"；
3. 数字、日期、编号必须与证据逐字一致，不要改写；
4. 回答用中文，简洁，一段话以内。"""


def run_agent(
    document: Document,
    question: str,
    settings: Settings,
    provider: LLMProvider | None = None,
    registry: ToolRegistry | None = None,
    max_steps: int | None = None,
) -> AgentResult:
    """运行一次工具调用式问答。

    Args:
        document: 已建索引的文档。
        question: 用户问题。
        settings: 配置（含 agent_max_steps）。
        provider: 模型提供方；None 时用 fake。
        registry: 工具集；None 时按文档构建。
        max_steps: 覆盖配置里的最大步数。

    Returns:
        AgentResult（含执行轨迹与用量）。
    """
    model = provider or create_provider(settings)
    tools = registry or build_document_tools(document, settings)
    limit = max_steps or settings.agent_max_steps

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=question),
    ]
    steps: list[AgentStep] = []
    evidence_ids: list[str] = []
    usage = Usage()

    for _ in range(limit):
        try:
            response = model.chat(messages, tools=tools.specs())
        except LLMError:
            logger.exception("模型调用失败，回退抽取式作答")
            return _fallback(document, question, settings)

        usage = Usage(
            prompt_tokens=usage.prompt_tokens + response.usage.prompt_tokens,
            completion_tokens=usage.completion_tokens + response.usage.completion_tokens,
            latency_ms=usage.latency_ms + response.usage.latency_ms,
        )
        message = response.message
        if not message.tool_calls:
            return _finalize(document, message.content, evidence_ids, steps, usage, settings)
        messages.append(message)
        for call in message.tool_calls:
            result = tools.call(call.name, call.arguments)
            steps.append(
                AgentStep(
                    kind="tool",
                    name=call.name,
                    arguments=dict(call.arguments),
                    result=result,
                )
            )
            evidence_ids.extend(_evidence_from(result))
            messages.append(
                ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=call.call_id,
                    name=call.name,
                )
            )

    return AgentResult(
        answer="",
        abstained=True,
        reasons=(f"超过最大步数 {limit}，未形成结论",),
        steps=tuple(steps),
        usage=usage,
    )


def _evidence_from(result: str) -> list[str]:
    """从工具结果里提取证据 chunk_id。

    Args:
        result: 工具返回的 JSON 字符串。

    Returns:
        chunk_id 列表。
    """
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return []
    return [
        str(item["chunk_id"])
        for item in payload.get("results", [])
        if isinstance(item, dict) and "chunk_id" in item
    ]


def _finalize(
    document: Document,
    answer: str,
    evidence_ids: list[str],
    steps: list[AgentStep],
    usage: Usage,
    settings: Settings,
) -> AgentResult:
    """收尾：生成引用、校验、决定是否弃答。

    Args:
        document: 文档。
        answer: 模型答案。
        evidence_ids: 证据 chunk_id（按出现顺序）。
        steps: 执行轨迹。
        usage: 用量。
        settings: 配置。

    Returns:
        AgentResult。
    """
    answer = (answer or "").strip()
    chunks = [document.chunk_by_id(chunk_id) for chunk_id in dict.fromkeys(evidence_ids)]
    valid_chunks = [chunk for chunk in chunks if chunk is not None]
    citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            region_id=chunk.region_id,
            page=chunk.page,
            bbox=chunk.bbox,
            snippet=chunk.text[:160],
            score=1.0,
        )
        for chunk in valid_chunks[:3]
    ]
    if not answer:
        return AgentResult(
            answer="",
            abstained=True,
            reasons=("模型没有给出答案",),
            steps=tuple(steps),
            usage=usage,
        )
    if not citations:
        return AgentResult(
            answer="",
            abstained=True,
            reasons=("没有工具证据支撑答案",),
            steps=tuple(steps),
            usage=usage,
        )
    verification = verify_answer(answer, citations, valid_chunks, min_support=0.35)
    if not verification.ok:
        return AgentResult(
            answer="",
            abstained=True,
            reasons=verification.reasons,
            steps=tuple(steps),
            usage=usage,
        )
    return AgentResult(
        answer=answer,
        citations=tuple(citations),
        steps=tuple(steps),
        usage=usage,
    )


def _fallback(document: Document, question: str, settings: Settings) -> AgentResult:
    """模型不可用时的降级：抽取式作答。

    Args:
        document: 文档。
        question: 问题。
        settings: 配置。

    Returns:
        AgentResult（backend 标记为 extractive）。
    """
    result = answer_question(document, question, settings)
    return AgentResult(
        answer=result.answer,
        citations=result.citations,
        abstained=result.abstained,
        reasons=result.reasons + ("模型不可用，已回退抽取式",),
        backend="extractive-fallback",
    )
