"""Fake 提供方：用脚本模拟模型行为，测试与演示都不需要联网。

两种用法：
1. **测试**：把预先设计好的响应排成队列，断言 Agent 是否正确执行工具、是否收敛；
2. **演示**：``FakeProvider.demo()`` 内置一个"先检索、再作答"的脚本，
   让没有 API key 的人也能跑通完整的工具调用链路。

这不是"假功能"：它验证的是编排逻辑（工具分发、消息拼接、终止条件），
真实模型接上后走同一条路径。
"""

from __future__ import annotations

import time

from visioncite.llm.base import ChatMessage, LLMResponse, ToolCall, Usage


class FakeProvider:
    """按脚本返回固定响应的提供方。"""

    name = "fake"

    def __init__(self, script: list[LLMResponse] | None = None) -> None:
        """初始化。

        Args:
            script: 预设响应列表；为空时使用 ``demo_script()``。
        """
        self._script = list(script) if script else demo_script()
        self._cursor = 0
        self.calls: list[list[ChatMessage]] = []

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """返回脚本中的下一条响应。

        Args:
            messages: 对话历史（会被记录，方便断言）。
            tools: 工具定义（fake 不使用，但保留签名一致）。

        Returns:
            LLMResponse；脚本用完后返回"证据不足"的兜底回答。
        """
        self.calls.append(list(messages))
        started = time.perf_counter()
        if self._cursor >= len(self._script):
            return LLMResponse(
                message=ChatMessage(role="assistant", content=""),
                usage=Usage(latency_ms=(time.perf_counter() - started) * 1000),
            )
        response = self._script[self._cursor]
        self._cursor += 1
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return LLMResponse(message=response.message, usage=usage)


def tool_call_response(name: str, arguments: dict[str, object], call_id: str = "call-1") -> LLMResponse:
    """构造一条"模型要求调用工具"的响应。

    Args:
        name: 工具名。
        arguments: 参数。
        call_id: 调用 id。

    Returns:
        LLMResponse。
    """
    return LLMResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(call_id=call_id, name=name, arguments=arguments),),
        ),
        usage=Usage(prompt_tokens=120, completion_tokens=18),
    )


def final_response(content: str) -> LLMResponse:
    """构造一条"模型给出最终答案"的响应。

    Args:
        content: 答案文本。

    Returns:
        LLMResponse。
    """
    return LLMResponse(
        message=ChatMessage(role="assistant", content=content),
        usage=Usage(prompt_tokens=260, completion_tokens=48),
    )


def demo_script() -> list[LLMResponse]:
    """演示脚本：先检索，再作答。

    Returns:
        两步响应列表。
    """
    return [
        tool_call_response("search_document", {"query": "合计金额", "top_k": 3}),
        final_response("合计金额：1,286.40 元。"),
    ]
