"""文档工具集：检索、取区域、看颜色、列页面。

这些工具把 VisionCite 已有的能力暴露给模型：
- ``search_document`` 复用 BM25 + CLIP + RRF 的检索；
- ``get_region`` / ``list_pages`` 让模型能核对证据位置；
- ``dominant_color`` 让模型能回答颜色问题（否则它只能瞎猜）。

每个工具都返回**结构化且带出处**的结果，Agent 才能据此生成带引用的答案。
"""

from __future__ import annotations

from visioncite.answer.color import dominant_color
from visioncite.config import Settings
from visioncite.index.retrieve import retrieve
from visioncite.pipeline import Document
from visioncite.tools.base import ToolRegistry, ToolSpec


def build_document_tools(document: Document, settings: Settings) -> ToolRegistry:
    """为一份文档构建工具集。

    Args:
        document: 已建索引的文档。
        settings: 配置（检索 top_k 等）。

    Returns:
        ToolRegistry。
    """
    registry = ToolRegistry()
    registry.register(_search_tool(document, settings))
    registry.register(_get_region_tool(document))
    registry.register(_list_pages_tool(document))
    registry.register(_color_tool(document))
    return registry


def _search_tool(document: Document, settings: Settings) -> ToolSpec:
    """构造检索工具。

    Args:
        document: 文档。
        settings: 配置。

    Returns:
        ToolSpec。
    """

    def handler(arguments: dict[str, object]) -> dict[str, object]:
        query = str(arguments.get("query", ""))
        top_k = int(arguments.get("top_k", 5))
        results = retrieve(document, query, settings)[: max(1, min(top_k, 10))]
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "page": item.chunk.page + 1,
                    "region_id": item.chunk.region_id,
                    "score": round(item.score, 4),
                    "text": item.chunk.text,
                }
                for item in results
            ],
        }

    return ToolSpec(
        name="search_document",
        description="在当前文档里检索与问题相关的段落，返回带 chunk_id、页码、分数的证据。回答前应先调用它。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词或问题"},
                "top_k": {"type": "integer", "description": "返回条数，默认 5，最大 10"},
            },
            "required": ["query"],
        },
        handler=handler,
    )


def _get_region_tool(document: Document) -> ToolSpec:
    """构造取区域工具。

    Args:
        document: 文档。

    Returns:
        ToolSpec。
    """

    def handler(arguments: dict[str, object]) -> dict[str, object]:
        region_id = str(arguments.get("region_id", ""))
        for page in document.pages:
            for region in page.regions:
                if region.region_id == region_id:
                    return {
                        "region_id": region.region_id,
                        "page": page.index + 1,
                        "kind": region.kind,
                        "bbox": region.bbox.to_list(),
                        "text": region.text,
                    }
        return {"error": f"找不到区域 {region_id}"}

    return ToolSpec(
        name="get_region",
        description="按 region_id 查看某个版面区域的原文、类型与坐标，用于核对证据。",
        parameters={
            "type": "object",
            "properties": {"region_id": {"type": "string", "description": "版面区域 id，如 p0-r3"}},
            "required": ["region_id"],
        },
        handler=handler,
    )


def _list_pages_tool(document: Document) -> ToolSpec:
    """构造列页面工具。

    Args:
        document: 文档。

    Returns:
        ToolSpec。
    """

    def handler(arguments: dict[str, object]) -> dict[str, object]:
        return {
            "pages": [
                {
                    "page": page.index + 1,
                    "width": page.width,
                    "height": page.height,
                    "regions": [
                        {"region_id": region.region_id, "kind": region.kind}
                        for region in page.regions
                    ],
                }
                for page in document.pages
            ]
        }

    return ToolSpec(
        name="list_pages",
        description="列出文档的页面与区域结构（不含正文），用于判断文档里有哪些页、哪些区域。",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def _color_tool(document: Document) -> ToolSpec:
    """构造颜色统计工具。

    Args:
        document: 文档。

    Returns:
        ToolSpec。
    """

    def handler(arguments: dict[str, object]) -> dict[str, object]:
        region_id = arguments.get("region_id")
        page_index = int(arguments.get("page", 1)) - 1
        page = document.page(page_index)
        if page is None:
            return {"error": f"页面不存在: {page_index + 1}"}
        region = None
        if region_id:
            region = next((item for item in page.regions if item.region_id == str(region_id)), None)
            if region is None:
                return {"error": f"找不到区域 {region_id}"}
        stat = dominant_color(page.image, region.bbox if region else None)
        if stat is None:
            return {"error": "该区域没有可统计的墨迹像素"}
        return {
            "page": page.index + 1,
            "region_id": region.region_id if region else None,
            "color": stat.name,
            "ratio": round(stat.ratio, 3),
        }

    return ToolSpec(
        name="dominant_color",
        description="统计某页/某区域的主色（黑/灰/红/蓝…），用于回答字体颜色、印章颜色这类问题。",
        parameters={
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "页码，从 1 开始"},
                "region_id": {"type": "string", "description": "可选，限定到某个区域"},
            },
            "required": ["page"],
        },
        handler=handler,
    )
