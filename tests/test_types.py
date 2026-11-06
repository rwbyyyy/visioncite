"""数据结构测试：BBox 是所有几何计算的基石，必须严谨。"""

from __future__ import annotations

import pytest

from visioncite.types import BBox, Chunk, Citation


def test_bbox_normalizes_reversed_coordinates() -> None:
    """传反的坐标要被自动纠正，避免出现负面积。"""
    box = BBox(100, 80, 20, 10)
    assert (box.x0, box.y0, box.x1, box.y1) == (20, 10, 100, 80)
    assert box.area == 80 * 70


def test_iou_identical_boxes_is_one() -> None:
    """完全重合时 IoU=1。"""
    box = BBox(0, 0, 10, 10)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    """不相交时 IoU=0。"""
    assert BBox(0, 0, 10, 10).iou(BBox(20, 20, 30, 30)) == 0.0


def test_iou_half_overlap() -> None:
    """横向错开一半：交集 50，并集 150，IoU=1/3。"""
    assert BBox(0, 0, 10, 10).iou(BBox(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_merge_covers_both_boxes() -> None:
    """合并后的框必须同时包含两个输入框。"""
    merged = BBox(0, 0, 10, 10).merge(BBox(20, 30, 40, 50))
    assert merged.to_list() == [0, 0, 40, 50]


def test_pad_clips_to_page_bounds() -> None:
    """扩张后要能被页面边界裁剪，避免越界。"""
    box = BBox(2, 2, 8, 8).pad(5, bounds=(10, 10))
    assert box.to_list() == [0, 0, 10, 10]


def test_from_list_rejects_wrong_length() -> None:
    """长度不是 4 时直接报错，防止静默错位。"""
    with pytest.raises(ValueError):
        BBox.from_list([1, 2, 3])


def test_chunk_is_hashable() -> None:
    """Chunk 是 frozenset 友好的不可变对象。"""
    chunk = Chunk(
        chunk_id="d:p0-r0:0",
        doc_id="d",
        page=0,
        region_id="p0-r0",
        bbox=BBox(0, 0, 1, 1),
        kind="text",
        text="x",
    )
    assert hash(chunk)


def test_citation_defaults() -> None:
    """Citation 的 score 有默认值，便于测试构造。"""
    citation = Citation(
        chunk_id="c",
        region_id="p0-r0",
        page=0,
        bbox=BBox(0, 0, 5, 5),
        snippet="测试",
    )
    assert citation.score == 0.0
