"""PDF 解析：把每一页渲染成图片，并尽量抽出文本层。

为什么以"渲染成图"为主：
    多模态问答的证据是"页面上的一块区域"，而文本层没有可靠坐标。
    统一走图片之后，扫描件和电子版 PDF 用同一条流水线，失败模式也更一致。
    文本层只作为 OCR 不可用时的降级来源。

依赖 pypdfium2：它是 PDFium 的 Python 绑定，wheel 自带二进制，
不需要系统装 poppler，Windows/Linux 行为一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray

from visioncite.errors import IngestError
from visioncite.types import Page


def count_pages(path: Path | str) -> int:
    """返回 PDF 页数。

    Args:
        path: PDF 路径。

    Returns:
        页数。

    Raises:
        IngestError: 文件无法打开（损坏、加密）。
    """
    try:
        document = pdfium.PdfDocument(path)
        try:
            return len(document)
        finally:
            document.close()
    except pdfium.PdfiumError as exc:
        msg = f"PDF 无法打开 {path}: {exc}"
        raise IngestError(msg) from exc


def render_pages(path: Path | str, dpi: int = 144, max_pages: int = 40) -> list[Page]:
    """把 PDF 每页渲染成 RGB 图片。

    Args:
        path: PDF 路径。
        dpi: 渲染分辨率；144 对 OCR 是常用的性价比点。
        max_pages: 最多渲染页数，防止一个超大 PDF 拖垮服务。

    Returns:
        页面列表，顺序与原文一致。

    Raises:
        IngestError: 打开或渲染失败。
    """
    scale = max(0.5, dpi / 72.0)  # PDF 的用户空间单位是 1/72 英寸
    try:
        document = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        msg = f"PDF 无法打开 {path}: {exc}"
        raise IngestError(msg) from exc

    pages: list[Page] = []
    try:
        for index in range(min(len(document), max_pages)):
            page = document[index]
            image = _render_one(page, scale)
            height, width = image.shape[:2]
            pages.append(
                Page(
                    index=index,
                    width=width,
                    height=height,
                    image=image,
                    text_layer=_extract_text_layer(page),
                )
            )
    finally:
        document.close()
    return pages


def _render_one(page: pdfium.PdfPage, scale: float) -> NDArray[np.uint8]:
    """渲染单页为 RGB 数组。

    Args:
        page: PDFium 页面对象。
        scale: 缩放比例（1.0 = 72dpi）。

    Returns:
        ``(H, W, 3)`` 数组。

    Raises:
        IngestError: 渲染失败。
    """
    try:
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()
    except Exception as exc:  # noqa: BLE001 - PDFium 抛出的类型不稳定，统一包装
        msg = f"PDF 第 {page.get_index() + 1} 页渲染失败: {exc}"
        raise IngestError(msg) from exc


def _extract_text_layer(page: pdfium.PdfPage) -> str:
    """尝试抽取文本层；扫描件会得到空串，这是正常情况。

    Args:
        page: PDFium 页面对象。

    Returns:
        该页文本；失败时返回空串（不阻断后续 OCR）。
    """
    try:
        textpage = page.get_textpage()
        return textpage.get_text_range().strip()
    except Exception:  # noqa: BLE001 - 文本层缺失不应让整份文档失败
        return ""
