"""MCP 协议测试：握手、工具发现、调用与错误处理。"""

from __future__ import annotations

from pathlib import Path

import pytest

from visioncite.config import Settings
from visioncite.index.visual import NullVisualEncoder
from visioncite.mcp.client import McpClient
from visioncite.mcp.server import McpServer
from visioncite.pipeline import Document, build_document
from visioncite.tools.document_tools import build_document_tools
from visioncite.types import Word


@pytest.fixture
def server(
    document_path: tuple[Path, list[Word]],
    fake_ocr,  # noqa: ANN001
    settings: Settings,
) -> McpServer:
    """用假 OCR 文档构建 MCP server。"""
    path, _ = document_path
    document: Document = build_document(path, settings, ocr=fake_ocr, visual=NullVisualEncoder())
    return McpServer(build_document_tools(document, settings))


def test_initialize_returns_capabilities(server: McpServer) -> None:
    """握手返回协议版本与 tools 能力。"""
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response is not None
    result = response["result"]
    assert result["protocolVersion"]
    assert "tools" in result["capabilities"]


def test_tools_list(server: McpServer) -> None:
    """tools/list 返回 name/description/inputSchema。"""
    response = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = response["result"]["tools"]  # type: ignore[index]
    assert {tool["name"] for tool in tools} == {
        "search_document",
        "get_region",
        "list_pages",
        "dominant_color",
    }
    assert all("inputSchema" in tool for tool in tools)


def test_tools_call_returns_content(server: McpServer) -> None:
    """tools/call 结果放在 content[0].text，且 isError=False。"""
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_document", "arguments": {"query": "金额"}},
        }
    )
    result = response["result"]  # type: ignore[index]
    assert result["isError"] is False
    assert "chunk_id" in result["content"][0]["text"]


def test_tools_call_unknown_tool_is_error(server: McpServer) -> None:
    """未知工具返回 isError=True，而不是协议错误。"""
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert response["result"]["isError"] is True  # type: ignore[index]


def test_unknown_method_returns_error(server: McpServer) -> None:
    """未实现的方法返回 JSON-RPC 错误码 -32601。"""
    response = server.handle({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
    assert response["error"]["code"] == -32601  # type: ignore[index]


def test_notification_gets_no_response(server: McpServer) -> None:
    """通知（无 id）不需要响应。"""
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_client_roundtrip(server: McpServer) -> None:
    """客户端能完成握手、发现工具、调用工具。"""
    client = McpClient.in_process(server)
    info = client.initialize()
    assert info["serverInfo"]["name"] == "visioncite"
    tool_names = [tool["name"] for tool in client.list_tools()]
    assert "search_document" in tool_names
    output = client.call_tool("search_document", {"query": "金额"})
    assert "chunk_id" in output
