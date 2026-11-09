"""工具层：把系统能力包装成模型可调用的函数。

设计要点：
1. **工具是纯函数**：输入参数 → JSON 结果，不读全局状态，便于单测与 MCP 复用；
2. **描述即文档**：``description`` 直接给模型看，写清楚"什么时候用、返回什么"；
3. **参数校验**：只校验 required 字段，类型错误由处理函数抛，统一转成 ``{"error": ...}``，
   让模型看到错误后可以自行重试，而不是让整个 Agent 崩掉。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from visioncite.logging import get_logger

logger = get_logger("tools")

ToolHandler = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的元信息与实现。

    Attributes:
        name: 工具名（模型看到的名字，建议 snake_case）。
        description: 给模型看的说明。
        parameters: JSON Schema 形式的参数定义。
        handler: 执行函数，返回可 JSON 序列化的 dict。
    """

    name: str
    description: str
    parameters: dict[str, object]
    handler: ToolHandler

    def to_openai(self) -> dict[str, object]:
        """转成 OpenAI function-calling 的工具定义。

        Returns:
            dict。
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolRegistry:
    """工具注册表：按名字分发调用。

    Attributes:
        _tools: 名字 → 工具定义。
    """

    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        """注册一个工具（同名覆盖，便于测试替换）。

        Args:
            spec: 工具定义。
        """
        self._tools[spec.name] = spec

    @property
    def names(self) -> list[str]:
        """已注册的工具名（排序后，保证输出稳定）。"""
        return sorted(self._tools)

    def specs(self) -> list[dict[str, object]]:
        """给模型用的工具定义列表。

        Returns:
            按名字排序的定义。
        """
        return [self._tools[name].to_openai() for name in self.names]

    def call(self, name: str, arguments: dict[str, object]) -> str:
        """执行工具并把结果序列化为字符串（tool 消息内容）。

        Args:
            name: 工具名。
            arguments: 参数。

        Returns:
            JSON 字符串；出错时是 ``{"error": "..."}``。
        """
        spec = self._tools.get(name)
        if spec is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        missing = _missing_required(spec.parameters, arguments)
        if missing:
            message = "缺少参数: " + ", ".join(missing)
            return json.dumps({"error": message}, ensure_ascii=False)
        try:
            return json.dumps(spec.handler(arguments), ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - 工具错误要回给模型，而不是中断 Agent
            logger.exception("工具执行失败: %s", name)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


def _missing_required(parameters: dict[str, object], arguments: dict[str, object]) -> list[str]:
    """检查必填参数。

    Args:
        parameters: JSON Schema。
        arguments: 实际参数。

    Returns:
        缺失的参数名列表。
    """
    required = parameters.get("required") or []
    return [str(name) for name in required if name not in arguments]  # type: ignore[union-attr]
