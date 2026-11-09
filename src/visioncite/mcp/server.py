"""MCP Server：把工具注册表暴露给任何 MCP 客户端。

``handle`` 是纯函数式入口（请求 → 响应），``serve`` 负责 stdio 循环；
这样协议逻辑可以脱离进程/管道单独测试。
"""

from __future__ import annotations

import json

from visioncite.logging import get_logger
from visioncite.mcp.protocol import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    StdioTransport,
    Transport,
    make_error,
    make_result,
)
from visioncite.tools.base import ToolRegistry

logger = get_logger("mcp.server")


class McpServer:
    """最小 MCP Server。"""

    def __init__(
        self,
        registry: ToolRegistry,
        name: str = "visioncite",
        version: str = "0.2.0",
    ) -> None:
        """初始化。

        Args:
            registry: 对外暴露的工具集。
            name: serverInfo.name。
            version: serverInfo.version。
        """
        self._registry = registry
        self._name = name
        self._version = version

    def handle(self, request: dict) -> dict | None:
        """处理一条 JSON-RPC 消息。

        Args:
            request: 请求消息。

        Returns:
            响应；通知（无 id）返回 None。
        """
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        if "id" not in request:  # 通知，不需要响应
            return None
        if method == "initialize":
            return make_result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self._name, "version": self._version},
                },
            )
        if method == "ping":
            return make_result(request_id, {})
        if method == "tools/list":
            return make_result(request_id, {"tools": self._tool_definitions()})
        if method == "tools/call":
            return self._call_tool(request_id, params)
        return make_error(request_id, METHOD_NOT_FOUND, f"未实现的方法: {method}")

    def _tool_definitions(self) -> list[dict[str, object]]:
        """把工具定义转成 MCP 格式（inputSchema）。

        Returns:
            工具定义列表。
        """
        return [
            {
                "name": item["name"],
                "description": item["description"],
                "inputSchema": item["parameters"],
            }
            for item in self._registry.specs()
        ]

    def _call_tool(self, request_id: object, params: dict) -> dict:
        """执行 tools/call。

        Args:
            request_id: 请求 id。
            params: ``{"name": ..., "arguments": {...}}``。

        Returns:
            响应。
        """
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return make_error(request_id, INVALID_PARAMS, "tools/call 需要 name 与 arguments")
        result = self._registry.call(name, arguments)
        is_error = result.startswith('{"error"')
        return make_result(
            request_id,
            {"content": [{"type": "text", "text": result}], "isError": is_error},
        )

    def serve(self, transport: Transport | None = None) -> None:
        """在传输上循环处理消息。

        Args:
            transport: 传输；None 时使用 stdin/stdout。
        """
        channel = transport or StdioTransport(reader=_stdin_line, writer=_stdout_line)
        while True:
            message = channel.receive()
            if message is None:
                return
            if message.get("method") == "__invalid__":
                channel.send(make_error(None, -32700, "JSON 解析失败"))
                continue
            try:
                response = self.handle(message)
            except Exception as exc:  # noqa: BLE001 - 单条消息失败不应终止服务
                logger.exception("处理 MCP 消息失败")
                response = make_error(message.get("id"), -32603, str(exc))
            if response is not None:
                channel.send(response)


def _stdin_line() -> str:
    """读取标准输入的一行。

    Returns:
        一行文本；EOF 返回空串。
    """
    import sys

    return sys.stdin.readline()


def _stdout_line(line: str) -> None:
    """写一行到标准输出并立即刷新。

    Args:
        line: 文本。
    """
    import sys

    sys.stdout.write(line)
    sys.stdout.flush()


def dumps_tool_result(payload: dict) -> str:
    """把工具结果编码成 MCP content 文本（给客户端复用）。

    Args:
        payload: 结果字典。

    Returns:
        JSON 字符串。
    """
    return json.dumps(payload, ensure_ascii=False)
