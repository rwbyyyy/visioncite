"""MCP 层：最小 Server / Client（stdio + JSON-RPC）。"""

from visioncite.mcp.client import McpClient
from visioncite.mcp.protocol import PROTOCOL_VERSION, StdioTransport, Transport
from visioncite.mcp.server import McpServer

__all__ = ["PROTOCOL_VERSION", "McpClient", "McpServer", "StdioTransport", "Transport"]
