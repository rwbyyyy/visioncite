"""MCP 的最小实现：JSON-RPC 2.0 + stdio 传输。

实现范围（明确写清楚，避免"看起来支持全协议"）：
- 支持 ``initialize`` / ``notifications/initialized`` / ``tools/list`` / ``tools/call`` / ``ping``；
- 只实现 **tools** 能力，不实现 resources、prompts、sampling、roots；
- 传输用 stdio（每行一个 JSON 消息），按 MCP 规范的消息分隔方式。

为什么手写而不装官方 SDK？
    协议本身很小：一个请求/响应 + 方法分发。手写一遍能看清
    "工具发现 → 调用 → 结果回传" 这条链路，也避免把项目绑在 SDK 版本上。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

PROTOCOL_VERSION = "2024-11-05"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class Transport(Protocol):
    """消息传输接口。"""

    def send(self, message: dict) -> None:
        """发送一条 JSON-RPC 消息。"""
        ...

    def receive(self) -> dict | None:
        """接收一条消息；EOF 返回 None。"""
        ...


class JsonRpcError(Exception):
    """带错误码的 JSON-RPC 异常。"""

    def __init__(self, code: int, message: str) -> None:
        """初始化。

        Args:
            code: JSON-RPC 错误码。
            message: 错误说明。
        """
        super().__init__(message)
        self.code = code
        self.message = message


def make_result(request_id: object, result: dict) -> dict:
    """构造成功响应。

    Args:
        request_id: 请求 id。
        result: 结果对象。

    Returns:
        JSON-RPC 响应。
    """
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: object, code: int, message: str) -> dict:
    """构造错误响应。

    Args:
        request_id: 请求 id（可能为 None）。
        code: 错误码。
        message: 错误说明。

    Returns:
        JSON-RPC 响应。
    """
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class StdioTransport:
    """基于文本流的传输（一行一条消息）。"""

    def __init__(self, reader: Callable[[], str], writer: Callable[[str], None]) -> None:
        """初始化。

        Args:
            reader: 读取一行的函数（返回空串表示 EOF）。
            writer: 写出一行的函数。
        """
        self._reader = reader
        self._writer = writer

    def send(self, message: dict) -> None:
        """发送消息。

        Args:
            message: JSON-RPC 消息。
        """
        self._writer(json.dumps(message, ensure_ascii=False) + "\n")

    def receive(self) -> dict | None:
        """接收消息。

        Returns:
            解析后的消息；EOF 或空行为 None。
        """
        line = self._reader()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "id": None, "method": "__invalid__", "params": {}}


class InProcessTransport:
    """进程内传输：把请求直接交给 server.handle，便于单测。

    Attributes:
        server: 处理消息的对象（需有 ``handle`` 方法）。
        outbox: 服务端发出的消息队列。
    """

    def __init__(self, server: object) -> None:
        """初始化。

        Args:
            server: 提供 ``handle(request) -> dict | None`` 的对象。
        """
        self._server = server
        self.outbox: list[dict] = []

    def send(self, message: dict) -> None:
        """发送请求并立即处理，把响应放进 outbox。

        Args:
            message: 请求消息。
        """
        response = self._server.handle(message)  # type: ignore[attr-defined]
        if response is not None:
            self.outbox.append(response)

    def receive(self) -> dict | None:
        """取出下一条响应。

        Returns:
            响应消息或 None。
        """
        return self.outbox.pop(0) if self.outbox else None
