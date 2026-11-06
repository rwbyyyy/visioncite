"""端到端测试：文件 → 版面 → 索引 → 问答（全部离线）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from visioncite.answer.qa import answer_question
from visioncite.config import Settings
from visioncite.errors import UnsupportedFileError
from visioncite.index.visual import NullVisualEncoder
from visioncite.ingest.loader import detect_kind
from visioncite.pipeline import build_document, document_id_from_bytes
from visioncite.types import Word


@pytest.fixture
def document(document_path: tuple[Path, list[Word]], fake_ocr, settings: Settings):  # noqa: ANN001, ANN201
    """用假 OCR 建好索引的文档。"""
    path, _words = document_path
    return build_document(path, settings, ocr=fake_ocr, visual=NullVisualEncoder())


def test_document_has_pages_and_chunks(document) -> None:  # noqa: ANN001
    """索引完成后应有页面与检索块。"""
    assert document.meta.page_count == 1
    assert document.chunks
    assert document.ocr_name == "fake"
    assert document.visual_name == "null"


def test_answer_question_cites_region(document, settings: Settings) -> None:
    """答案要带引用，且引用落在页面内。"""
    result = answer_question(document, "合计金额是多少？", settings)
    assert not result.abstained
    assert "1286.40" in result.answer.replace(",", "")
    assert result.citations
    citation = result.citations[0]
    assert citation.page == 0
    assert 0 <= citation.bbox.x0 < citation.bbox.x1 <= 640


def test_unanswerable_question_abstains(document, settings: Settings) -> None:
    """文档里没有的信息必须弃答。"""
    result = answer_question(document, "税额是多少？", settings)
    assert result.abstained
    assert result.answer == ""
    assert result.reasons


def test_empty_question_abstains(document, settings: Settings) -> None:
    """空问题直接弃答。"""
    result = answer_question(document, "   ", settings)
    assert result.abstained


def test_document_id_is_content_addressed() -> None:
    """同样的字节产生同样的 doc_id。"""
    assert document_id_from_bytes(b"abc") == document_id_from_bytes(b"abc")
    assert document_id_from_bytes(b"abc") != document_id_from_bytes(b"abd")


def test_detect_kind_by_magic_bytes(tmp_path: Path) -> None:
    """伪造扩展名也要按内容判断。"""
    fake_png = tmp_path / "looks_like.pdf"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    assert detect_kind(fake_png) == "image"


def test_detect_kind_rejects_unknown(tmp_path: Path) -> None:
    """未知类型要显式报错，而不是静默当图片处理。"""
    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsupportedFileError):
        detect_kind(target)
