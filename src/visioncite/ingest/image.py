"""图片读取与预处理。

任何多模态系统的第一步都是把"输入"变成统一的数组格式：
RGB、uint8、HWC 排布。统一之后，后面的版面、OCR、CLIP 都不必再关心来源。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from visioncite.errors import IngestError
from visioncite.types import BBox


def load_image(path: Path | str, max_side: int = 2400) -> NDArray[np.uint8]:
    """读取图片并规范化为 RGB 数组。

    处理了三件容易踩坑的事：
    1. EXIF 旋转（手机拍的照片方向经常是反的）；
    2. 调色板/灰度/CMYK 统一转 RGB；
    3. 超大图按最长边等比缩小，避免 OCR 内存爆炸。

    Args:
        path: 图片路径。
        max_side: 最长边上限（像素），<=0 表示不缩放。

    Returns:
        形状 ``(H, W, 3)``、dtype=uint8 的数组。

    Raises:
        IngestError: 文件无法解码。
    """
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw)
            image = image.convert("RGB")
            if max_side > 0:
                image = _shrink_to_fit(image, max_side)
            return np.asarray(image, dtype=np.uint8).copy()
    except OSError as exc:  # PIL 解码失败统一抛这里
        msg = f"无法读取图片 {path}: {exc}"
        raise IngestError(msg) from exc


def _shrink_to_fit(image: Image.Image, max_side: int) -> Image.Image:
    """等比缩小到最长边不超过 ``max_side``。

    Args:
        image: 原图。
        max_side: 允许的最长边。

    Returns:
        可能被缩小的新图；本来就够小则原样返回。
    """
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    # LANCZOS 比默认的 BICUBIC 更适合缩小，文字边缘更干净
    return image.resize(new_size, Image.LANCZOS)


def crop_image(image: NDArray[np.uint8], bbox: BBox) -> NDArray[np.uint8]:
    """按 bbox 裁剪数组，越界部分自动夹到图像内。

    Args:
        image: 源图，``(H, W, 3)``。
        bbox: 裁剪区域。

    Returns:
        裁剪结果；区域完全在图像外时返回 1x1 的空白图，避免下游崩溃。
    """
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, bbox.x0))
    y0 = max(0, min(height - 1, bbox.y0))
    x1 = max(x0 + 1, min(width, bbox.x1))
    y1 = max(y0 + 1, min(height, bbox.y1))
    return image[y0:y1, x0:x1].copy()


def to_png_bytes(image: NDArray[np.uint8]) -> bytes:
    """把数组编码为 PNG 字节流，供 API 直接返回。

    Args:
        image: ``(H, W, 3)`` 数组。

    Returns:
        PNG 字节。
    """
    import io

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def perceptual_hash(image: NDArray[np.uint8], size: int = 8) -> int:
    """差值哈希（dHash），用于快速判重。

    原理：把图缩成 ``(size+1) x size`` 的灰度图，比较相邻像素亮度，
    亮为 1 暗为 0，得到 ``size*size`` 位整数。
    相比像素比对，它对亮度、轻微缩放不敏感，成本又远低于 CLIP 向量。

    Args:
        image: 输入图像。
        size: 哈希边长，8 表示 64 位。

    Returns:
        整数哈希值。
    """
    gray = Image.fromarray(image).convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hash_distance(a: int, b: int) -> int:
    """两个哈希值的汉明距离（不同的位数）。

    Args:
        a: 哈希值 A。
        b: 哈希值 B。

    Returns:
        不同位的个数；越小越相似，一般 <=5 视为同一画面。
    """
    return bin(a ^ b).count("1")
