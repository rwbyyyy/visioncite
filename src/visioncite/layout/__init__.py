"""版面分析：二值化 → XY-cut → 表格线 → 区域装配。"""

from visioncite.layout.binarize import ink_density, ink_mask, otsu_threshold, to_gray
from visioncite.layout.regions import build_regions, classify, group_rows, reading_order
from visioncite.layout.table import find_tables, rule_bands
from visioncite.layout.xycut import CutConfig, xy_cut

__all__ = [
    "CutConfig",
    "build_regions",
    "classify",
    "find_tables",
    "group_rows",
    "ink_density",
    "ink_mask",
    "otsu_threshold",
    "reading_order",
    "rule_bands",
    "to_gray",
    "xy_cut",
]
