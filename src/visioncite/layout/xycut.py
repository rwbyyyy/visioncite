"""递归 XY-cut 版面分割。

算法（经典文档分析做法，面试可以手画）：
1. 对当前区域求**水平投影**：每一行有多少墨迹像素；
2. 投影为 0 的行是"空白带"；空白带足够宽（>= min_gap）就切开；
3. 切成若干横条后，对每条再求**垂直投影**，按空白列竖着切；
4. 横竖交替递归，直到切不动、区域太小或达到深度上限。

为什么不用深度学习版面模型？
    XY-cut 无参数、CPU 上毫秒级、每一刀都能画出来解释，
    对"规则排版的报告/票据/幻灯片"足够好。失效场景也很明确：
    多栏但栏间不留白、背景有噪点、倾斜扫描 —— 写进 LIMITATIONS.md。

复杂度：每层递归对区域像素做一次投影，最深 D 层，
    总计 O(D * 页面像素)；144dpi 的一页实测几毫秒。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from visioncite.types import BBox

# 叶子块的最小墨迹占比：低于它的块基本是噪点，直接丢弃
_MIN_INK = 0.002


@dataclass(frozen=True)
class CutConfig:
    """切分参数。

    Attributes:
        min_gap: 空白带最小宽度（像素），小于它的间隙不当作分隔。
        min_size: 内容块最小边长，太小丢弃（噪点、标点）。
        max_depth: 递归深度上限，防止病态页面无限切分。
    """

    min_gap: int = 12
    min_size: int = 24
    max_depth: int = 6


def xy_cut(mask: NDArray[np.bool_], config: CutConfig | None = None) -> list[BBox]:
    """对整页墨迹做递归 XY-cut。

    Args:
        mask: ``(H, W)`` 的布尔墨迹矩阵。
        config: 切分参数，None 用默认值。

    Returns:
        内容块列表，顺序大致是从上到下、从左到右。
    """
    cfg = config or CutConfig()
    height, width = mask.shape[:2]
    blocks: list[BBox] = []
    _cut(mask, BBox(0, 0, width, height), depth=0, axis=0, cfg=cfg, out=blocks)
    return blocks


def _cut(
    mask: NDArray[np.bool_],
    box: BBox,
    depth: int,
    axis: int,
    cfg: CutConfig,
    out: list[BBox],
) -> None:
    """递归处理一个区域。

    Args:
        mask: 整页墨迹矩阵。
        box: 当前区域（整页坐标）。
        depth: 当前深度。
        axis: 0 表示按行投影（横切），1 表示按列投影（竖切）。
        cfg: 参数。
        out: 收集叶子块。
    """
    if depth >= cfg.max_depth or box.width < cfg.min_size or box.height < cfg.min_size:
        _append_leaf(mask, box, out)
        return

    piece = mask[box.y0 : box.y1, box.x0 : box.x1]
    segments = _split_segments(piece, axis=axis, min_gap=cfg.min_gap)
    if len(segments) > 1:
        for start, end in segments:
            child = _child_box(box, start, end, axis)
            _cut(mask, child, depth + 1, 1 - axis, cfg, out)
        return

    # 当前方向切不动，换一个方向再试一次（多栏文档常见：先横切失败、竖切有效）
    alt_axis = 1 - axis
    alt_segments = _split_segments(piece, axis=alt_axis, min_gap=cfg.min_gap)
    if len(alt_segments) > 1:
        for start, end in alt_segments:
            child = _child_box(box, start, end, alt_axis)
            _cut(mask, child, depth + 1, axis, cfg, out)
        return

    _append_leaf(mask, box, out)


def _child_box(parent: BBox, start: int, end: int, axis: int) -> BBox:
    """把"沿某方向的一段"换算成子区域。

    Args:
        parent: 父区域。
        start: 段起点（相对父区域）。
        end: 段终点（相对父区域）。
        axis: 0 表示段在 y 方向，1 表示段在 x 方向。

    Returns:
        子区域。
    """
    if axis == 0:
        return BBox(parent.x0, parent.y0 + start, parent.x1, parent.y0 + end)
    return BBox(parent.x0 + start, parent.y0, parent.x0 + end, parent.y1)


def _split_segments(
    piece: NDArray[np.bool_],
    axis: int,
    min_gap: int,
) -> list[tuple[int, int]]:
    """按投影找内容段。

    先求投影（axis=0 求每行墨迹数，axis=1 求每列墨迹数），
    把"有墨迹"的连续区间视为内容段；间距小于 ``min_gap`` 的段合并，
    因为它们视觉上属于同一块（例如两行字之间的行距）。

    Args:
        piece: 区域墨迹矩阵。
        axis: 0 按行，1 按列。
        min_gap: 合并阈值。

    Returns:
        ``[(start, end), ...]``，若无有效切分则返回长度为 1 的列表。
    """
    profile = piece.sum(axis=1 if axis == 0 else 0)
    occupied = profile > 0
    runs = _true_runs(occupied)
    if not runs:
        return []

    merged: list[list[int]] = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] < min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _true_runs(flags: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """找布尔序列中连续 True 的区间。

    技巧：首尾各补一个 False，再做差分，
    差分 = +1 的位置是区间起点，-1 的位置是终点（左闭右开）。

    Args:
        flags: 一维布尔数组。

    Returns:
        区间列表。
    """
    padded = np.concatenate(([False], flags, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _append_leaf(mask: NDArray[np.bool_], box: BBox, out: list[BBox]) -> None:
    """收尾：墨迹太少的块当噪点丢弃。

    Args:
        mask: 整页墨迹矩阵。
        box: 待判定的区域。
        out: 结果列表。
    """
    if box.area <= 0:
        return
    patch = mask[box.y0 : box.y1, box.x0 : box.x1]
    if patch.size == 0 or patch.mean() < _MIN_INK:
        return
    out.append(box)
