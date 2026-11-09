"""MCP 命令行：``python -m visioncite.mcp serve <doc>`` 等。

用途：
- ``serve``：把某份文档的工具暴露成 MCP 服务，供 Inspector / 其他宿主连接；
- ``list``：一次性列出工具，验证服务是否正常；
- ``call``：一次性调用某个工具，便于调试参数。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from visioncite.config import Settings
from visioncite.errors import VisionCiteError
from visioncite.logging import get_logger, setup_logging
from visioncite.mcp.client import McpClient
from visioncite.mcp.server import McpServer
from visioncite.pipeline import build_document
from visioncite.tools.document_tools import build_document_tools

logger = get_logger("mcp")


def _build(path: str) -> tuple[object, Settings]:
    """解析文档并构建工具。

    Args:
        path: 文档路径。

    Returns:
        ``(document, settings)``。
    """
    settings = Settings.from_env()
    settings.ensure_dirs()
    document = build_document(Path(path), settings)
    return document, settings


def _serve(args: argparse.Namespace) -> int:
    """启动 stdio MCP 服务。

    Args:
        args: 命令行参数。

    Returns:
        退出码。
    """
    document, settings = _build(args.path)
    server = McpServer(build_document_tools(document, settings))
    logger.info("MCP 服务就绪，工具: %s", ", ".join(server._registry.names))  # noqa: SLF001
    server.serve()
    return 0


def _list(args: argparse.Namespace) -> int:
    """列出工具。

    Args:
        args: 命令行参数。

    Returns:
        退出码。
    """
    document, settings = _build(args.path)
    client = McpClient.in_process(McpServer(build_document_tools(document, settings)))
    print(json.dumps(client.list_tools(), ensure_ascii=False, indent=2))
    return 0


def _call(args: argparse.Namespace) -> int:
    """调用一个工具。

    Args:
        args: 命令行参数。

    Returns:
        退出码。
    """
    document, settings = _build(args.path)
    client = McpClient.in_process(McpServer(build_document_tools(document, settings)))
    arguments = json.loads(args.arguments or "{}")
    print(client.call_tool(args.tool, arguments))
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口。

    Args:
        argv: 参数列表。

    Returns:
        退出码。
    """
    setup_logging()
    parser = argparse.ArgumentParser(prog="visioncite.mcp", description="VisionCite MCP 服务")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 stdio MCP 服务")
    serve.add_argument("path")
    serve.set_defaults(handler=_serve)

    listing = sub.add_parser("list", help="列出工具")
    listing.add_argument("path")
    listing.set_defaults(handler=_list)

    call = sub.add_parser("call", help="调用工具")
    call.add_argument("path")
    call.add_argument("tool")
    call.add_argument("arguments", nargs="?", default="{}")
    call.set_defaults(handler=_call)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except VisionCiteError as exc:
        print(f"失败：{exc.user_message}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
