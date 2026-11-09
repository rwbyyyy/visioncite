"""LLM 抽象层测试：fake 脚本、OpenAI 兼容解析、缺少 key 的处理。"""

from __future__ import annotations

import pytest

from visioncite.errors import LLMError
from visioncite.llm.base import ChatMessage
from visioncite.llm.fake import FakeProvider, final_response, tool_call_response
from visioncite.llm.openai_compat import OpenAICompatProvider


def test_fake_provider_follows_script() -> None:
    """fake 按顺序返回脚本，并记录调用历史。"""
    provider = FakeProvider(
        [
            tool_call_response("search_document", {"query": "金额"}),
            final_response("好的"),
        ]
    )
    first = provider.chat([ChatMessage(role="user", content="Q")])
    second = provider.chat([ChatMessage(role="user", content="Q")])
    assert first.message.tool_calls[0].name == "search_document"
    assert second.message.content == "好的"
    assert len(provider.calls) == 2


def test_fake_provider_exhausted_returns_empty() -> None:
    """脚本用完后返回空答案，而不是抛异常。"""
    provider = FakeProvider([final_response("唯一一条")])
    provider.chat([])
    assert provider.chat([]).message.content == ""


class _FakeResponse:
    """httpx.Response 的最小替身。"""

    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


def test_openai_provider_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """兼容层能把 tool_calls 解析成内部 ToolCall。"""
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-9",
                            "type": "function",
                            "function": {"name": "search_document", "arguments": '{"query": "金额"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
    }
    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _FakeResponse(body))
    provider = OpenAICompatProvider("https://example.com/v1", "test-model", api_key="k")
    response = provider.chat([ChatMessage(role="user", content="Q")], tools=[{"name": "t"}])
    assert response.message.tool_calls[0].call_id == "call-9"
    assert response.message.tool_calls[0].arguments == {"query": "金额"}
    assert response.usage.total_tokens == 16


def test_openai_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有 API key 时显式报 LLMError，而不是发一个必然失败的请求。"""
    monkeypatch.delenv("VISIONCITE_LLM_API_KEY", raising=False)
    provider = OpenAICompatProvider("https://example.com/v1", "m", api_key=None)
    with pytest.raises(LLMError):
        provider.chat([ChatMessage(role="user", content="Q")])
