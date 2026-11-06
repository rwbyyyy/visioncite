"""颜色统计测试：合成纯色图，验证主色判定与背景排除。"""

from __future__ import annotations

import numpy as np

from visioncite.answer.color import dominant_color
from visioncite.types import BBox


def _canvas(width: int = 200, height: int = 80, color: tuple[int, int, int] = (255, 255, 255)):
    """生成纯色画布。"""
    return np.full((height, width, 3), color, dtype=np.uint8)


def test_black_text_on_white_is_black() -> None:
    """白底黑字 → 主色黑色。"""
    image = _canvas()
    image[20:30, 20:180] = 0
    image[45:55, 20:180] = 0
    stat = dominant_color(image)
    assert stat is not None
    assert stat.name == "黑色"
    assert stat.ratio > 0.9


def test_blue_block_is_blue() -> None:
    """蓝色块 → 蓝色。"""
    image = _canvas()
    image[20:60, 20:180] = (30, 70, 220)
    stat = dominant_color(image)
    assert stat is not None
    assert stat.name == "蓝色"


def test_red_block_is_red() -> None:
    """红色块 → 红色（印章场景）。"""
    image = _canvas()
    image[10:70, 10:190] = (200, 20, 20)
    stat = dominant_color(image)
    assert stat is not None
    assert stat.name == "红色"


def test_blank_background_returns_none() -> None:
    """纯白页没有墨迹像素 → 返回 None，而不是瞎猜颜色。"""
    assert dominant_color(_canvas()) is None


def test_light_watermark_is_ignored() -> None:
    """浅色水印不能算作墨迹。"""
    image = _canvas(color=(232, 245, 233))
    assert dominant_color(image) is None


def test_bbox_restricts_statistics() -> None:
    """只统计指定区域：左下角蓝色块不影响左上角黑字区域。"""
    image = _canvas(400, 200)
    image[10:40, 10:180] = 0
    image[120:190, 200:390] = (20, 60, 220)
    stat = dominant_color(image, BBox(0, 0, 190, 50))
    assert stat is not None
    assert stat.name == "黑色"


def test_mixed_ink_reports_dominant_with_ratio() -> None:
    """黑红混合时，比例高的颜色胜出且 ratio 合理。"""
    image = _canvas(200, 100)
    image[0:60, :] = 0
    image[60:80, :] = (200, 20, 20)
    stat = dominant_color(image)
    assert stat is not None
    assert stat.name == "黑色"
    assert 0.5 < stat.ratio < 1.0
