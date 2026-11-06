"""文档处理流水线：把"一个文件"变成"可检索的 Document"。

这是唯一知道全部步骤的地方（编排层），每一步都只做一件事：
    load_pages → OCR → 版面区域 → 切块 → BM25 → 视觉向量

为什么把编排单独拿出来？
    如果 OCR 直接调用版面、版面直接调用 BM25，模块之间就纠缠在一起，
    测试只能整体跑。现在的写法允许每一步单独注入 fake 实现：
        build_document(path, settings, ocr=FakeOcr(), visual=NullVisual())
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from visioncite.config import Settings
from visioncite.errors import VisualUnavailableError
from visioncite.index.bm25 import BM25Index
from visioncite.index.chunking import chunks_from_pages
from visioncite.index.visual import ClipVisualEncoder, NullVisualEncoder, create_visual
from visioncite.ingest.image import crop_image
from visioncite.ingest.loader import load_pages
from visioncite.layout.regions import build_regions
from visioncite.logging import get_logger
from visioncite.ocr.base import OcrBackend
from visioncite.ocr.factory import create_ocr
from visioncite.types import BBox, Chunk, DocumentMeta, Page, Region, Word

logger = get_logger("pipeline")

VisualEncoder = ClipVisualEncoder | NullVisualEncoder


@dataclass
class Document:
    """一份已建好索引的文档（API 层用它做问答）。

    Attributes:
        meta: 元信息。
        pages: 页面（含图片，用于证据裁图）。
        chunks: 检索块。
        bm25: 文本索引。
        vectors: 与 ``chunks`` 对齐的视觉向量；未启用视觉时为 None。
        ocr_name: 实际使用的 OCR 后端名。
        visual_name: 实际使用的视觉后端名。
    """

    meta: DocumentMeta
    pages: list[Page]
    chunks: list[Chunk]
    bm25: BM25Index
    vectors: NDArray[np.float32] | None = None
    ocr_name: str = "null"
    visual_name: str = "null"
    # 编码器实例只在内存里用，不参与序列化，因此 repr=False
    encoder: VisualEncoder | None = field(default=None, repr=False)
    _chunk_index: dict[str, int] = field(default_factory=dict, repr=False)

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """按 id 取块。

        Args:
            chunk_id: 块 id。

        Returns:
            块或 None。
        """
        if not self._chunk_index:
            self._chunk_index = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}
        index = self._chunk_index.get(chunk_id)
        return self.chunks[index] if index is not None else None

    def page(self, index: int) -> Page | None:
        """按页码取页面。

        Args:
            index: 页码，从 0 开始。

        Returns:
            页面或 None。
        """
        if 0 <= index < len(self.pages):
            return self.pages[index]
        return None


def document_id_from_bytes(data: bytes) -> str:
    """用内容指纹做 doc_id：同一文件重复上传会命中同一个 id。

    Args:
        data: 文件字节。

    Returns:
        16 位十六进制字符串。
    """
    return hashlib.sha256(data).hexdigest()[:16]


def build_document(
    path: Path | str,
    settings: Settings,
    ocr: OcrBackend | None = None,
    visual: VisualEncoder | None = None,
    doc_id: str | None = None,
) -> Document:
    """完整处理一份文档。

    Args:
        path: 文件路径。
        settings: 运行配置。
        ocr: 注入的 OCR 后端；None 表示按配置创建。
        visual: 注入的视觉编码器；None 表示按配置创建。
        doc_id: 指定文档 id；None 用文件内容哈希。

    Returns:
        建好索引的 Document。

    Raises:
        UnsupportedFileError: 文件类型不支持。
        IngestError: 解析失败。
    """
    file_path = Path(path)
    pages = load_pages(file_path, settings)
    backend = ocr or create_ocr(settings)
    encoder = visual or create_visual(settings)

    for page in pages:
        _analyze_page(page, backend, settings)

    chunks = chunks_from_pages(doc_id or _fingerprint(file_path), pages)
    bm25 = BM25Index(settings.bm25_k1, settings.bm25_b)
    bm25.fit([(chunk.chunk_id, chunk.text) for chunk in chunks])
    vectors = _encode_chunks(pages, chunks, encoder)

    meta = DocumentMeta(
        doc_id=doc_id or _fingerprint(file_path),
        filename=file_path.name,
        page_count=len(pages),
        size_bytes=file_path.stat().st_size,
    )
    logger.info(
        "文档索引完成 doc=%s pages=%d chunks=%d vectors=%s",
        meta.doc_id,
        len(pages),
        len(chunks),
        "on" if vectors is not None else "off",
    )
    return Document(
        meta=meta,
        pages=pages,
        chunks=chunks,
        bm25=bm25,
        vectors=vectors,
        ocr_name=backend.name,
        visual_name=encoder.name,
        encoder=encoder,
    )


def _analyze_page(page: Page, ocr: OcrBackend, settings: Settings) -> None:
    """识别并切分单页区域，结果写回 ``page.regions``。

    Args:
        page: 页面对象（原地修改）。
        ocr: OCR 后端。
        settings: 配置。
    """
    words = ocr.recognize(page.image)
    if not words and page.text_layer:
        # 扫描件/纯图页面没有字，但 PDF 可能带文本层：退化成整页一个文本区域
        page.regions = [_text_layer_region(page)]
        return
    page.regions = build_regions(page.index, page.image, words)


def _text_layer_region(page: Page) -> Region:
    """把 PDF 文本层包装成覆盖整页的区域。

    Args:
        page: 页面。

    Returns:
        整页文本区域。
    """
    return Region(
        region_id=f"p{page.index}-r0",
        page=page.index,
        bbox=BBox(0, 0, page.width, page.height),
        kind="text",
        text=page.text_layer,
        words=[],
        confidence=1.0,
    )


def _encode_chunks(
    pages: list[Page],
    chunks: list[Chunk],
    encoder: VisualEncoder,
) -> NDArray[np.float32] | None:
    """为每个块计算视觉向量（按 region 去重）。

    Args:
        pages: 页面。
        chunks: 块。
        encoder: 视觉编码器。

    Returns:
        ``(N, D)`` 向量；不可用时返回 None。
    """
    if not encoder.available() or not chunks:
        return None
    try:
        crops: list[NDArray[np.uint8]] = []
        region_to_vector: dict[tuple[int, str], int] = {}
        chunk_to_vector: list[int] = []
        for chunk in chunks:
            key = (chunk.page, chunk.region_id)  # 同一区域可能被切成多块，只编码一次
            if key not in region_to_vector:
                page = pages[chunk.page]
                region_to_vector[key] = len(crops)
                crops.append(
                    crop_image(page.image, chunk.bbox.pad(4, (page.width, page.height)))
                )
            chunk_to_vector.append(region_to_vector[key])
        vectors = encoder.encode_images(crops)
        return np.stack([vectors[index] for index in chunk_to_vector])
    except VisualUnavailableError:
        return None
    except Exception:  # noqa: BLE001 - 编码失败不应阻断文本检索
        logger.exception("视觉编码失败，本次仅使用文本检索")
        return None


def _fingerprint(path: Path) -> str:
    """文件的 sha256 指纹（前 16 位）。

    Args:
        path: 文件路径。

    Returns:
        十六进制 id。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def make_word(text: str, x0: int, y0: int, x1: int, y1: int, confidence: float = 1.0) -> Word:
    """构造 Word 的快捷函数（测试与合成语料常用）。

    Args:
        text: 文本。
        x0: 左。
        y0: 上。
        x1: 右。
        y1: 下。
        confidence: 置信度。

    Returns:
        Word 对象。
    """
    return Word(text=text, bbox=BBox(x0, y0, x1, y1), confidence=confidence)
