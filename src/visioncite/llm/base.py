"""LLM 抽象层：消息、工具调用、用量。

为什么先做抽象？
    项目默认是"不依赖大模型"的抽取式基线。要加 Agent / 工具调用，
    就不能把某个厂商 SDK 写死进业务代码，否则测试必须联网、必须带 key。
    这里用 Protocol 定义最小接口，fake 与真实实现可互换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用。

    Attributes:
        call_id: 调用 id，回填工具结果时要用它对齐。
        name: 工具名。
        arguments: 参数（已解析为 dict）。
    """

    call_id: str
    name: str
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    """一条对话消息（兼容 OpenAI 的 tool 消息语义）。

    Attributes:
        role: system / user / assistant / tool。
        content: 文本内容。
        tool_calls: assistant 消息里模型提出的工具调用。
        tool_call_id: tool 消息对应的调用 id。
        name: tool 消息的工具名。
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class Usage:
    """一次调用的开销。

    Attributes:
        prompt_tokens: 输入 token 数。
        completion_tokens: 输出 token 数。
        latency_ms: 客户端计时。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 数。"""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResponse:
    """一次模型调用结果。

    Attributes:
        message: 模型返回的消息。
        usage: 用量。
    """

    message: ChatMessage
    usage: Usage = field(default_factory=Usage)


class LLMProvider(Protocol):
    """模型提供方接口（结构化子类型）。"""

    name: str

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """执行一次对话（可选携带工具定义）。"""
        ...
