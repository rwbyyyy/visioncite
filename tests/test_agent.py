"""Agent 循环测试：工具调用、引用生成、弃答与降级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from visioncite.agent.loop import run_agent
from visioncite.config import Settings
from visioncite.errors import LLMError
from visioncite.index.visual import NullVisualEncoder
from visioncite.llm.base import ChatMessage, LLMResponse
from visioncite.llm.fake import FakeProvider, final_response, tool_call_response
from visioncite.pipeline import Document, build_document
from visioncite.types import Word


@pytest.fixture
def document(
    document_path: tuple[Path, list[Word]],
    fake_ocr,  # noqa: ANN001
    settings: Settings,
) -> Document:
    """用假 OCR 建一份文档。"""
    path, _ = document_path
    return build_document(path, settings, ocr=fake_ocr, visual=NullVisualEncoder())


def test_agent_calls_tool_then_answers(document: Document, settings: Settings) -> None:
    """先检索再作答：轨迹里应有工具调用，答案带引用。"""
    provider = FakeProvider(
        [
            tool_call_response("search_document", {"query": "合计金额", "top_k": 3}),
            final_response("合计金额为 1,286.40 元。"),
        ]
    )
    result = run_agent(document, "合计金额是多少？", settings, provider=provider)
    assert not result.abstained
    assert "1,286.40" in result.answer
    assert result.steps and result.steps[0].name == "search_document"
    assert result.citations
    assert result.usage.total_tokens > 0


def test_agent_abstains_without_tool_evidence(document: Document, settings: Settings) -> None:
    """模型没查资料就作答 → 弃答（没有证据支撑）。"""
    provider = FakeProvider([final_response("合计金额是 999 元。")])
    result = run_agent(document, "合计金额是多少？", settings, provider=provider)
    assert result.abstained
    assert "证据" in "".join(result.reasons)


def test_agent_stops_at_max_steps(document: Document, settings: Settings) -> None:
    """模型反复调工具时，按最大步数终止。"""
    loop_call = tool_call_response("search_document", {"query": "金额"})
    provider = FakeProvider([loop_call] * 5)
    result = run_agent(document, "金额？", settings, provider=provider, max_steps=2)
    assert result.abstained
    assert "步数" in "".join(result.reasons)


def test_agent_falls_back_when_model_fails(document: Document, settings: Settings) -> None:
    """模型报错时回退抽取式，而不是把异常抛给用户。"""

    class BrokenProvider:
        name = "broken"

        def chat(self, messages: list[ChatMessage], tools: list[dict] | None = None) -> LLMResponse:
            raise LLMError("网络断了")

    result = run_agent(document, "合计金额是多少？", settings, provider=BrokenProvider())
    assert result.backend == "extractive-fallback"
    assert "1286.40" in result.answer
