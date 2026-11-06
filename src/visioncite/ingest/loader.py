"""统一入口：根据文件内容判断类型，产出 Page 列表。

设计要点：
1. **按魔数（magic bytes）判断**，而不是只看扩展名 —— 扩展名可以随便改；
2. 判断逻辑集中在 ``detect_kind``，新增格式只改这一处；
3. 不在这里做 OCR/版面，只负责"把文件变成图"，单一职责。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from visioncite.config import Settings
from visioncite.errors import IngestError, UnsupportedFileError
from visioncite.ingest.image import load_image
from visioncite.ingest.pdf import render_pages
from visioncite.types import Page

FileKind = Literal["pdf", "image"]

# 文件头 → 类型。只列我们真正支持的格式，避免"能识别不能处理"的假承诺。
_MAGIC_PREFIXES: dict[bytes, FileKind] = {
    b"%PDF": "pdf",
    b"\x89PNG\r\n\x1a\n": "image",
    b"\xff\xd8\xff": "image",  # JPEG
    b"GIF87a": "image",
    b"GIF89a": "image",
    b"RIFF": "image",  # WebP 也以 RIFF 开头，PIL 会进一步校验
}
_SUFFIXES: dict[str, FileKind] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
}


def detect_kind(path: Path | str) -> FileKind:
    """判断文件类型：先看魔数，再看扩展名。

    Args:
        path: 文件路径。

    Returns:
        ``"pdf"`` 或 ``"image"``。

    Raises:
        UnsupportedFileError: 两者都无法识别。
    """
    file_path = Path(path)
    with file_path.open("rb") as handle:
        head = handle.read(16)
    for prefix, kind in _MAGIC_PREFIXES.items():
        if head.startswith(prefix):
            # RIFF 需要再排除非图片（例如 .wav），交给文件名兜底判断
            if prefix == b"RIFF" and file_path.suffix.lower() not in _SUFFIXES:
                continue
            return kind
    suffix_kind = _SUFFIXES.get(file_path.suffix.lower())
    if suffix_kind is None:
        msg = f"不支持的文件类型: {file_path.suffix or '(无扩展名)'}"
        raise UnsupportedFileError(msg)
    return suffix_kind


def load_pages(path: Path | str, settings: Settings) -> list[Page]:
    """把文件解析成页面列表。

    Args:
        path: 文件路径。
        settings: 运行配置（渲染 DPI、图片缩放上限等）。

    Returns:
        页面列表；图片文件固定为 1 页。

    Raises:
        UnsupportedFileError: 类型不支持。
        IngestError: 解析失败。
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"文件不存在: {file_path}"
        raise IngestError(msg)

    kind = detect_kind(file_path)
    if kind == "pdf":
        return render_pages(file_path, dpi=settings.render_dpi)

    image = load_image(file_path, max_side=settings.max_image_side)
    height, width = image.shape[:2]
    return [Page(index=0, width=width, height=height, image=image)]
