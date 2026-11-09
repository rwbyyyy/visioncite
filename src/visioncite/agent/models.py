"""Agent 结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from visioncite.llm.base import Usage
from visioncite.types import Citation


@dataclass(frozen=True)
class AgentStep:
    """Agent 执行的一步。

    Attributes:
        kind: 目前只有 ``tool``（模型调用工具）。
        name: 工具名。
        arguments: 调用参数。
        result: 工具返回的 JSON 字符串。
    """

    kind: str
    name: str
    arguments: dict[str, object] = field(default_factory=dict)
    result: str = ""


@dataclass(frozen=True)
class AgentResult:
    """Agent 的最终结果。

    Attributes:
        answer: 答案文本；弃答为空。
        citations: 证据引用。
        steps: 执行轨迹（可用于前端展示"它查了什么"）。
        abstained: 是否弃答。
        reasons: 弃答/降级原因。
        usage: token 与耗时。
        backend: ``agent`` 或 ``extractive-fallback``。
    """

    answer: str
    citations: tuple[Citation, ...] = ()
    steps: tuple[AgentStep, ...] = ()
    abstained: bool = False
    reasons: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)
    backend: str = "agent"
