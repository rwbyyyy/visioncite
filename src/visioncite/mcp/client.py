"""MCP Client：发现工具并调用。

既支持进程内（测试用 InProcessTransport），也支持通过 stdio 启动真实子进程
（``python -m visioncite.mcp serve <doc>``），用于对接 MCP Inspector 等外部宿主。
"""

from __future__ import annotations

import json
import subprocess
import sys

from visioncite.mcp.protocol import InProcessTransport, StdioTransport, Transport
from visioncite.mcp.server import McpServer


class McpClient:
    """最小 MCP 客户端。"""

    def __init__(self, transport: Transport) -> None:
        """初始化。

        Args:
            transport: 传输实现。
        """
        self._transport = transport
        self._next_id = 0
        self.server_info: dict[str, object] = {}

    @classmethod
    def in_process(cls, server: McpServer) -> McpClient:
        """构造进程内客户端（单测用）。

        Args:
            server: 服务端实例。

        Returns:
            McpClient。
        """
        return cls(InProcessTransport(server))

    @classmethod
    def spawn(cls, document_path: str) -> McpClient:
        """启动一个真实的 MCP 子进程（stdio）。

        Args:
            document_path: 交给 server 的文档路径。

        Returns:
            McpClient。

        Raises:
            RuntimeError: 子进程无法启动。
        """
        process = subprocess.Popen(  # noqa: S603 - 参数来自本进程，非用户输入
            [sys.executable, "-m", "visioncite.mcp", "serve", document_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        if process.stdin is None or process.stdout is None:  # pragma: no cover
            msg = "无法启动 MCP 子进程"
            raise RuntimeError(msg)
        transport = StdioTransport(
            reader=lambda: process.stdout.readline() if process.stdout else "",
            writer=lambda line: process.stdin.write(line) if process.stdin else None,
        )
        return cls(transport)

    def request(self, method: str, params: dict | None = None) -> dict:
        """发送请求并等待响应。

        Args:
            method: 方法名。
            params: 参数。

        Returns:
            响应里的 result。

        Raises:
            RuntimeError: 传输断开或返回错误。
        """
        self._next_id += 1
        self._transport.send(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        )
        response = self._transport.receive()
        if response is None:
            msg = "MCP 连接已关闭"
            raise RuntimeError(msg)
        if "error" in response:
            raise RuntimeError(f"MCP 错误: {response['error']}")
        return response.get("result") or {}

    def initialize(self) -> dict:
        """执行握手。

        Returns:
            initialize 结果。
        """
        result = self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
        self.server_info = result.get("serverInfo") or {}
        return result

    def list_tools(self) -> list[dict]:
        """列出工具。

        Returns:
            工具定义列表。
        """
        return self.request("tools/list").get("tools") or []

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """调用工具并返回文本结果。

        Args:
            name: 工具名。
            arguments: 参数。

        Returns:
            content 里的第一段文本。
        """
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        content = result.get("content") or []
        if not content:
            return ""
        return str(content[0].get("text", ""))


def format_tool_call(name: str, arguments: dict, output: str) -> str:
    """把一次调用格式化成可读文本（示例与调试用）。

    Args:
        name: 工具名。
        arguments: 参数。
        output: 结果文本。

    Returns:
        多行字符串。
    """
    return json.dumps(
        {"tool": name, "arguments": arguments, "output": json.loads(output or "null")},
        ensure_ascii=False,
        indent=2,
    )
