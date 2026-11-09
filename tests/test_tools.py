"""工具层测试：检索、取区域、列页面、颜色、参数校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visioncite.config import Settings
from visioncite.index.visual import NullVisualEncoder
from visioncite.pipeline import Document, build_document
from visioncite.tools.base import ToolRegistry, ToolSpec
from visioncite.tools.document_tools import build_document_tools
from visioncite.types import Word


@pytest.fixture
def document(
    document_path: tuple[Path, list[Word]],
    fake_ocr,  # noqa: ANN001
    settings: Settings,
) -> Document:
    """用假 OCR 建一份文档。"""
    path, _ = document_path
    return build_document(path, settings, ocr=fake_ocr, visual=NullVisualEncoder())


def test_registry_lists_sorted_tools(document: Document, settings: Settings) -> None:
    """工具列表稳定有序，定义符合 OpenAI 格式。"""
    registry = build_document_tools(document, settings)
    assert registry.names == ["dominant_color", "get_region", "list_pages", "search_document"]
    spec = registry.specs()[0]
    assert {"name", "description", "parameters"} <= set(spec)


def test_registry_rejects_missing_arguments(document: Document, settings: Settings) -> None:
    """缺少必填参数时返回 error，而不是抛异常。"""
    registry = build_document_tools(document, settings)
    payload = json.loads(registry.call("search_document", {}))
    assert "error" in payload


def test_registry_unknown_tool(document: Document, settings: Settings) -> None:
    """未知工具返回 error。"""
    registry = build_document_tools(document, settings)
    assert "error" in json.loads(registry.call("nope", {}))


def test_registry_wraps_handler_exception() -> None:
    """处理函数抛异常时转成 error，Agent 可以继续。"""

    def boom(arguments: dict[str, object]) -> dict[str, object]:
        raise ValueError("坏参数")

    registry = ToolRegistry()
    registry.register(ToolSpec(name="boom", description="d", parameters={"type": "object"}, handler=boom))
    payload = json.loads(registry.call("boom", {}))
    assert "ValueError" in payload["error"]


def test_search_document_returns_evidence(document: Document, settings: Settings) -> None:
    """检索工具返回 chunk_id 与文本，供模型引用。"""
    registry = build_document_tools(document, settings)
    payload = json.loads(registry.call("search_document", {"query": "合计金额", "top_k": 3}))
    assert payload["results"]
    first = payload["results"][0]
    assert {"chunk_id", "page", "text", "score"} <= set(first)


def test_get_region_returns_bbox(document: Document, settings: Settings) -> None:
    """区域工具返回坐标，便于核对证据位置。"""
    registry = build_document_tools(document, settings)
    region_id = document.pages[0].regions[0].region_id
    payload = json.loads(registry.call("get_region", {"region_id": region_id}))
    assert payload["region_id"] == region_id
    assert len(payload["bbox"]) == 4


def test_list_pages_shape(document: Document, settings: Settings) -> None:
    """页面工具返回页数与区域类型。"""
    registry = build_document_tools(document, settings)
    payload = json.loads(registry.call("list_pages", {}))
    assert payload["pages"][0]["page"] == 1
    assert "regions" in payload["pages"][0]


def test_dominant_color_tool(document: Document, settings: Settings) -> None:
    """颜色工具能返回主色（合成图是黑块）。"""
    registry = build_document_tools(document, settings)
    payload = json.loads(registry.call("dominant_color", {"page": 1}))
    assert payload["color"] in {"黑色", "灰色"}
