"""LLM 层：提供方接口、fake 实现、OpenAI 兼容客户端。"""

from visioncite.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    ToolCall,
    Usage,
)
from visioncite.llm.factory import create_provider
from visioncite.llm.fake import FakeProvider, final_response, tool_call_response
from visioncite.llm.openai_compat import OpenAICompatProvider

__all__ = [
    "ChatMessage",
    "FakeProvider",
    "LLMProvider",
    "LLMResponse",
    "OpenAICompatProvider",
    "ToolCall",
    "Usage",
    "create_provider",
    "final_response",
    "tool_call_response",
]
