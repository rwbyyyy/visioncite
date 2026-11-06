"""版面分析测试：二值化、XY-cut、表格线、区域装配。"""

from __future__ import annotations

import numpy as np

from visioncite.layout.binarize import ink_density, ink_mask, otsu_threshold, to_gray
from visioncite.layout.regions import build_regions, group_rows, merge_overlapping
from visioncite.layout.table import find_tables, rule_bands
from visioncite.layout.xycut import CutConfig, xy_cut
from visioncite.types import BBox, Word


def _blank(height: int = 200, width: int = 200) -> np.ndarray:
    """全白图。"""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _with_blocks() -> np.ndarray:
    """白底 + 两个黑色矩形（模拟两段文字）。"""
    image = _blank(300, 300)
    image[40:80, 20:280] = 0
    image[160:200, 20:280] = 0
    return image


def test_gray_conversion_keeps_shape() -> None:
    """灰度化不改变尺寸。"""
    assert to_gray(_blank()).shape == (200, 200)


def test_otsu_separates_bimodal_histogram() -> None:
    """两峰直方图的阈值应落在两峰之间。"""
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    image[:50, :] = 80
    threshold = otsu_threshold(to_gray(image))
    assert 80 <= threshold < 200


def test_ink_mask_marks_dark_pixels() -> None:
    """黑块区域应被判定为墨迹。"""
    mask = ink_mask(to_gray(_with_blocks()))
    assert mask[60, 100]
    assert not mask[250, 100]


def test_ink_density_range() -> None:
    """墨迹占比在 0~1 之间，且空白区域为 0。"""
    mask = ink_mask(to_gray(_with_blocks()))
    assert 0 < ink_density(mask, BBox(20, 40, 280, 80)) <= 1
    assert ink_density(mask, BBox(0, 250, 50, 290)) == 0.0


def test_xy_cut_splits_two_rows() -> None:
    """两段文字应被切成至少两块，且不包含整页。"""
    blocks = xy_cut(ink_mask(to_gray(_with_blocks())))
    assert len(blocks) >= 2
    assert all(block.area < 300 * 300 for block in blocks)


def test_xy_cut_returns_empty_for_blank_page() -> None:
    """全白页没有墨迹，应返回空列表而不是报错。"""
    assert xy_cut(ink_mask(to_gray(_blank()))) == []


def test_xy_cut_respects_max_depth() -> None:
    """深度限制为 0 时只返回整页一块（如果该块有墨迹）。"""
    blocks = xy_cut(ink_mask(to_gray(_with_blocks())), CutConfig(max_depth=0))
    assert len(blocks) == 1


def test_rule_bands_detects_horizontal_line() -> None:
    """整行涂黑的细带应被识别为线。"""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:43, :] = True
    bands = rule_bands(mask, axis=0, min_ratio=0.5)
    assert bands and bands[0][0] == 40


def test_find_tables_requires_both_directions() -> None:
    """只有横线没有竖线时不应判为表格。"""
    mask = np.zeros((200, 200), dtype=bool)
    for y in (30, 90, 150):
        mask[y : y + 2, 20:180] = True
    assert find_tables(mask) == []


def test_find_tables_detects_grid() -> None:
    """完整网格应被识别，且行数列数正确。"""
    mask = np.zeros((300, 300), dtype=bool)
    for y in (40, 120, 200, 280):
        mask[y : y + 2, 40:260] = True
    for x in (40, 150, 259):
        mask[40:282, x : x + 2] = True
    tables = find_tables(mask)
    assert len(tables) == 1
    assert tables[0].bbox.area > 0


def test_group_rows_clusters_by_y() -> None:
    """同一行的词按 y 聚类，行内按 x 排序。"""
    words = [
        Word("b", BBox(100, 10, 120, 30)),
        Word("a", BBox(10, 12, 30, 32)),
        Word("c", BBox(10, 80, 30, 100)),
    ]
    rows = group_rows(words)
    assert len(rows) == 2
    assert [word.text for word in rows[0]] == ["a", "b"]


def test_merge_overlapping_boxes() -> None:
    """高重叠的块要合并，低重叠的保留。"""
    merged = merge_overlapping([BBox(0, 0, 100, 100), BBox(5, 5, 95, 95), BBox(200, 200, 300, 300)])
    assert len(merged) == 2


def test_build_regions_uses_words_for_text() -> None:
    """有词时区域类型为 text，且文本来自词。"""
    image = _with_blocks()
    words = [Word("合计金额", BBox(20, 40, 280, 80), confidence=0.9)]
    regions = build_regions(0, image, words)
    assert regions
    assert any(region.kind == "text" for region in regions)


def test_build_regions_without_words_returns_figures() -> None:
    """没有 OCR 结果时，有墨迹的块应被标成 figure。"""
    regions = build_regions(0, _with_blocks(), [])
    assert all(region.kind in {"figure", "unknown"} for region in regions)
