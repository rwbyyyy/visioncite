"""表格检测：用"贯穿全页的长直线"找带框线的表格。

思路（形态学/投影的简化版）：
1. 表格的横线在**行方向**上墨迹占比很高，竖线在**列方向**上占比很高；
2. 把占比超过阈值的行/列抽出来，就是横线/竖线；
3. 当横线 >= 3 条、竖线 >= 2 条时，用它们围成的矩形作为表格候选。

局限（必须知道，面试会被问）：
- 只支持**有可见框线**的表格；无线表（三线表只剩表头线、纯留白对齐表）检测不到；
- 整页只有一张大表时最准；多张小表会被包成一个大框；
- 倾斜 1~2 度以上，直线会断成多段。

这些限制不做隐藏，``docs/LIMITATIONS.md`` 与消融实验里都会写。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from visioncite.types import BBox


@dataclass(frozen=True)
class TableCandidate:
    """表格候选。

    Attributes:
        bbox: 表格外接矩形。
        rows: 估计行数（横线数 - 1）。
        cols: 估计列数（竖线数 - 1）。
    """

    bbox: BBox
    rows: int
    cols: int


def rule_bands(
    mask: NDArray[np.bool_],
    axis: int,
    min_ratio: float = 0.25,
) -> list[tuple[int, int]]:
    """找出"像一条线"的行/列带。

    Args:
        mask: 布尔墨迹矩阵。
        axis: 0 表示找横线（按行统计），1 表示找竖线（按列统计）。
        min_ratio: 该行/列的墨迹占比阈值。

    Returns:
        连续的线带 ``[(start, end), ...]``，一个带代表一条粗线或细线。
    """
    profile = mask.mean(axis=1 if axis == 0 else 0)
    strong = profile >= min_ratio
    runs = _binary_runs(strong)
    # 太厚的带（例如整块深色图片）不是线，过滤掉
    return [(s, e) for s, e in runs if (e - s) <= 12]


def find_tables(mask: NDArray[np.bool_], min_ratio: float = 0.25) -> list[TableCandidate]:
    """在整页里找表格。

    Args:
        mask: 布尔墨迹矩阵。
        min_ratio: 线判定阈值。

    Returns:
        表格候选列表；找不到时返回空列表。
    """
    horizontal = rule_bands(mask, axis=0, min_ratio=min_ratio)
    vertical = rule_bands(mask, axis=1, min_ratio=min_ratio)
    if len(horizontal) < 3 or len(vertical) < 2:
        return []

    x0 = vertical[0][0]
    x1 = vertical[-1][1]
    y0 = horizontal[0][0]
    y1 = horizontal[-1][1]
    if x1 - x0 < 40 or y1 - y0 < 30:  # 太小的"表格"多半误判
        return []
    return [TableCandidate(bbox=BBox(x0, y0, x1, y1), rows=len(horizontal) - 1, cols=len(vertical) - 1)]


def _binary_runs(flags: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """找连续 True 区间（与 xycut 里同一套技巧，但这里保持模块独立）。

    Args:
        flags: 一维布尔数组。

    Returns:
        区间列表（左闭右开）。
    """
    padded = np.concatenate(([False], flags, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]
