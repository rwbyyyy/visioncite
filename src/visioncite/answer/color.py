"""颜色属性问答：靠像素统计回答"这段字是什么颜色"。

为什么需要单独一条路径？
    OCR 只给"文字 + 位置"，不保留颜色；CLIP 也不擅长判断字色。
    但"字体颜色 / 底色"这类问题在票据、证书、报表场景很常见，
    用几十行像素统计就能稳定回答 —— 比调 VLM 更快、更可解释。

做法：
    1. 裁剪目标区域并降采样；
    2. 用 HSV 找出"墨迹像素"（足够暗或足够有彩色），排除白底与浅色水印；
    3. 按色相分桶（黑/灰/红/橙/黄/绿/青/蓝/紫/粉），统计占比；
    4. 返回占比最高的颜色名与比例。

边界（写在 docs/LIMITATIONS.md）：
    - 回答的是**区域主色**，不是每个字的颜色；
    - 渐变、半透明、强反光会拉低比例；
    - 不做语义判断（"这个红是不是公章"需要版面知识，当前没有）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from visioncite.types import BBox


@dataclass(frozen=True)
class ColorStat:
    """区域主色统计结果。

    Attributes:
        name: 中文颜色名。
        ratio: 该颜色占"墨迹像素"的比例（0~1）。
        rgb: 该颜色的代表 RGB（取同类像素均值）。
        ink_pixels: 参与统计的墨迹像素数。
    """

    name: str
    ratio: float
    rgb: tuple[int, int, int]
    ink_pixels: int


# 色相区间（单位：度）。区间按常见颜色命名习惯划分，够用且容易解释。
_HUE_BUCKETS: list[tuple[float, float, str]] = [
    (0, 15, "红色"),
    (15, 40, "橙色"),
    (40, 70, "黄色"),
    (70, 170, "绿色"),
    (170, 200, "青色"),
    (200, 260, "蓝色"),
    (260, 320, "紫色"),
    (320, 345, "粉色"),
    (345, 360, "红色"),
]


def dominant_color(
    image: NDArray[np.uint8],
    bbox: BBox | None = None,
    max_side: int = 480,
) -> ColorStat | None:
    """统计区域主色。

    Args:
        image: 页面 RGB 数组。
        bbox: 目标区域；None 表示整页。
        max_side: 降采样上限，控制耗时。

    Returns:
        ColorStat；区域内没有墨迹像素时返回 None。
    """
    crop = image if bbox is None else _crop(image, bbox)
    if crop.size == 0:
        return None
    crop = _downscale(crop, max_side)

    hue, sat, val = _to_hsv(crop)
    ink = _ink_mask(sat, val)
    if ink.sum() < 20:  # 太少像素的统计不可信
        return None

    names = _classify(hue, sat, val, ink)
    ink_names = names[ink]  # 只统计墨迹像素，避免 None 参与排序
    unique, counts = np.unique(ink_names, return_counts=True)
    best = int(np.argmax(counts))
    color_name = str(unique[best])
    mask = names == color_name
    rgb = tuple(int(v) for v in crop[mask].mean(axis=0))
    return ColorStat(
        name=color_name,
        ratio=float(counts[best] / ink.sum()),
        rgb=rgb,  # type: ignore[arg-type]
        ink_pixels=int(ink.sum()),
    )


def dominant_background(
    image: NDArray[np.uint8],
    bbox: BBox | None = None,
    max_side: int = 480,
) -> ColorStat | None:
    """统计区域"底色"（浅色背景）。

    与 :func:`dominant_color` 相反：这里只统计**亮且低饱和**的像素，
    用于回答"这张纸是什么底色"。无水印的纯白纸会得到"白色"。

    Args:
        image: 页面 RGB 数组。
        bbox: 目标区域；None 表示整页。
        max_side: 降采样上限。

    Returns:
        ColorStat；没有足够背景像素时返回 None。
    """
    crop = image if bbox is None else _crop(image, bbox)
    if crop.size == 0:
        return None
    crop = _downscale(crop, max_side)
    hue, sat, val = _to_hsv(crop)
    mask = (val > 0.80) & (sat < 0.30)
    if mask.sum() < 50:
        return None

    names = _background_names(hue, sat, val, mask)
    values = names[mask]
    unique, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    name = str(unique[best])
    rgb = tuple(int(v) for v in crop[mask & (names == name)].mean(axis=0))
    return ColorStat(
        name=name,
        ratio=float(counts[best] / mask.sum()),
        rgb=rgb,  # type: ignore[arg-type]
        ink_pixels=int(mask.sum()),
    )


def _background_names(
    hue: NDArray[np.float32],
    sat: NDArray[np.float32],
    val: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> NDArray[np.object_]:
    """给背景像素命名（白 / 浅灰 / 浅<色相>）。

    Args:
        hue: 色相（度）。
        sat: 饱和度。
        val: 明度。
        mask: 背景像素掩码。

    Returns:
        名称数组（非背景为 None）。
    """
    names: NDArray[np.object_] = np.full(hue.shape, None, dtype=object)
    names[mask & (val > 0.93) & (sat < 0.07)] = "白色"
    names[mask & (sat < 0.12) & ~(val > 0.93)] = "浅灰色"
    for low, high, name in _HUE_BUCKETS:
        in_bucket = mask & (sat >= 0.12) & (hue >= low) & (hue < high)
        names[in_bucket] = "浅" + name
    _fill_unclassified(names, mask, "浅灰色")
    return names


def _fill_unclassified(
    names: NDArray[np.object_],
    mask: NDArray[np.bool_],
    fallback: str,
) -> None:
    """给掩码内仍未命中的像素兜底命名，避免 None 混进统计。

    Args:
        names: 名称数组（原地修改）。
        mask: 参与统计的像素。
        fallback: 兜底名称。
    """
    is_none = np.frompyfunc(lambda value: value is None, 1, 1)(names).astype(bool)
    names[mask & is_none] = fallback


def find_red_bbox(image: NDArray[np.uint8], max_side: int = 900) -> BBox | None:
    """找出图中红色像素的外接框（用于定位印章/红章）。

    印章通常是整页里唯一的红色块；与其统计"整页主色"（会被黑字压过去），
    不如先框出红色区域，再只统计这一块。

    Args:
        image: 页面 RGB 数组。
        max_side: 降采样上限（坐标会换算回原图）。

    Returns:
        红色区域外接框；红色像素太少时返回 None。
    """
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    small = _downscale(image, max_side) if scale < 1.0 else image
    hue, sat, val = _to_hsv(small)
    red = ((hue < 18) | (hue >= 340)) & (sat > 0.45) & (val > 0.20)
    if red.sum() < 40:
        return None
    ys, xs = np.nonzero(red)
    inverse = 1.0 / scale
    return BBox(
        int(xs.min() * inverse),
        int(ys.min() * inverse),
        int((xs.max() + 1) * inverse),
        int((ys.max() + 1) * inverse),
    )


def _crop(image: NDArray[np.uint8], bbox: BBox) -> NDArray[np.uint8]:
    """按 bbox 裁剪并夹到图像范围内。

    Args:
        image: 页面图。
        bbox: 区域。

    Returns:
        裁剪结果。
    """
    height, width = image.shape[:2]
    x0 = max(0, min(width, bbox.x0))
    x1 = max(0, min(width, bbox.x1))
    y0 = max(0, min(height, bbox.y0))
    y1 = max(0, min(height, bbox.y1))
    return image[y0:y1, x0:x1]


def _downscale(image: NDArray[np.uint8], max_side: int) -> NDArray[np.uint8]:
    """等比缩小到最长边不超过 max_side。

    Args:
        image: 输入图。
        max_side: 上限。

    Returns:
        缩小后的数组（可能原样返回）。
    """
    longest = max(image.shape[:2])
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    pil = Image.fromarray(image).resize(new_size, Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _to_hsv(image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], ...]:
    """RGB → HSV（H 用度，S/V 用 0~1）。

    Args:
        image: RGB 数组。

    Returns:
        ``(hue, sat, val)`` 三个同形状数组。
    """
    hsv = np.asarray(Image.fromarray(image).convert("HSV"), dtype=np.float32)
    return hsv[..., 0] * (360.0 / 256.0), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def _ink_mask(sat: NDArray[np.float32], val: NDArray[np.float32]) -> NDArray[np.bool_]:
    """判定"墨迹像素"：足够暗（黑灰字）或足够有彩色（彩字、印章），且不是白底。

    Args:
        sat: 饱和度 0~1。
        val: 明度 0~1。

    Returns:
        布尔掩码。
    """
    dark = val < 0.72
    colorful = sat > 0.30
    background = (sat < 0.12) & (val > 0.86)
    return (dark | colorful) & ~background


def _classify(
    hue: NDArray[np.float32],
    sat: NDArray[np.float32],
    val: NDArray[np.float32],
    ink: NDArray[np.bool_],
) -> NDArray[np.object_]:
    """给每个墨迹像素分配颜色名。

    Args:
        hue: 色相（度）。
        sat: 饱和度。
        val: 明度。
        ink: 墨迹掩码。

    Returns:
        与掩码同形状的字符串数组（非墨迹位置为 None）。
    """
    names: NDArray[np.object_] = np.full(hue.shape, None, dtype=object)
    # 与 _ink_mask 的 colorful 阈值保持一致，保证"墨迹像素"都能被归类
    achromatic = ink & (sat < 0.30)
    names[achromatic & (val < 0.35)] = "黑色"
    names[achromatic & (val >= 0.35)] = "灰色"

    for low, high, name in _HUE_BUCKETS:
        in_bucket = ink & ~achromatic & (hue >= low) & (hue < high)
        names[in_bucket] = name
    _fill_unclassified(names, ink, "灰色")
    return names
