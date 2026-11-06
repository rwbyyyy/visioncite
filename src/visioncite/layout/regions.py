"""把"墨迹块 + OCR 词"装配成有语义的区域（Region）。

这一步承上启下：
- 向上接受 XY-cut 切出的几何块；
- 向下产出带 ``kind``（正文/表格/图）和 ``text`` 的区域，供检索使用。

分类规则刻意保持"可解释、可调试"：
    1. 与表格候选重叠超过一半 → table；
    2. 块里几乎没有识别到字、但有明显墨迹 → figure（图表、照片、印章）；
    3. 其余 → text。
复杂场景（公式、页眉页脚、多栏混排）不做过度工程，交给 LIMITATIONS 说明。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from visioncite.layout.binarize import ink_density, ink_mask, to_gray
from visioncite.layout.table import TableCandidate, find_tables
from visioncite.layout.xycut import CutConfig, xy_cut
from visioncite.types import BBox, Region, RegionKind, Word


def build_regions(
    page_index: int,
    image: NDArray[np.uint8],
    words: list[Word],
    cut_config: CutConfig | None = None,
) -> list[Region]:
    """生成一页的区域列表。

    Args:
        page_index: 页码。
        image: 页面 RGB 图。
        words: OCR 结果（可能为空）。
        cut_config: XY-cut 参数。

    Returns:
        按阅读顺序排序的区域列表。
    """
    mask = ink_mask(to_gray(image))
    tables = find_tables(mask)
    boxes = xy_cut(mask, cut_config)
    boxes = merge_overlapping(boxes)
    boxes = _drop_tiny(boxes)

    # 兜底：图片太干净（或纯色）时 XY-cut 切不出块，但 OCR 可能仍识别到字
    if not boxes:
        if not words:
            return []
        boxes = [BBox(0, 0, image.shape[1], image.shape[0])]

    grouped = assign_words(boxes, words)
    regions: list[Region] = []
    for index, box in enumerate(boxes):
        region_words = grouped[index]
        kind = classify(box, region_words, tables, mask)
        text = build_text(box, region_words, kind)
        confidence = _mean_confidence(region_words)
        regions.append(
            Region(
                region_id=f"p{page_index}-r{index}",
                page=page_index,
                bbox=tighten_bbox(box, region_words),
                kind=kind,
                text=text,
                words=region_words,
                confidence=confidence,
            )
        )
    return reading_order(regions)


def tighten_bbox(box: BBox, words: list[Word], padding: int = 4) -> BBox:
    """把区域框收紧到"实际文字"的范围。

    XY-cut 返回的是"整条内容带"，可能包含大段空白；
    直接拿它当证据框，前端高亮会显得很空，评测时 IoU 也会偏低。
    收紧之后：框住文字外接矩形，再向外留一点边距。

    Args:
        box: XY-cut 的几何块。
        words: 块内识别到的词。
        padding: 向外扩张的像素。

    Returns:
        收紧后的区域；块内无文字时原样返回。
    """
    if not words:
        return box
    x0 = min(word.bbox.x0 for word in words)
    y0 = min(word.bbox.y0 for word in words)
    x1 = max(word.bbox.x1 for word in words)
    y1 = max(word.bbox.y1 for word in words)
    return BBox(x0, y0, x1, y1).pad(padding)


def classify(
    box: BBox,
    words: list[Word],
    tables: list[TableCandidate],
    mask: NDArray[np.bool_],
) -> RegionKind:
    """判断区域类型。

    Args:
        box: 区域。
        words: 落在区域内的词。
        tables: 本页表格候选。
        mask: 墨迹矩阵。

    Returns:
        ``text`` / ``table`` / ``figure`` / ``unknown``。
    """
    for table in tables:
        inter = box.intersection_area(table.bbox)
        if box.area and inter / box.area > 0.5:
            return "table"
    if not words:
        return "figure" if ink_density(mask, box) > 0.05 else "unknown"
    return "text"


def build_text(box: BBox, words: list[Word], kind: RegionKind) -> str:
    """按区域类型拼文本。

    Args:
        box: 区域（表格时用于分列，暂未使用，保留扩展点）。
        words: 区域内的词。
        kind: 区域类型。

    Returns:
        文本；表格用 ``a | b`` 表示同一行的多个单元格。
    """
    if not words:
        return ""
    rows = group_rows(words)
    if kind == "table":
        return "\n".join(" | ".join(word.text for word in row) for row in rows)
    return "\n".join(" ".join(word.text for word in row) for row in rows)


def group_rows(words: list[Word], y_tolerance: float = 0.6) -> list[list[Word]]:
    """把词按行聚类。

    判定"同一行"的标准：两个词的垂直中心差 < 平均字高 * tolerance。
    这是最简单的版面行聚类，比按 y 排序更稳（避免上下浮动的下标被打散）。

    Args:
        words: 词列表。
        y_tolerance: 容差系数。

    Returns:
        行列表，每行内按 x 排序。
    """
    if not words:
        return []
    heights = [max(1, word.bbox.height) for word in words]
    tolerance = float(np.median(heights)) * y_tolerance

    ordered = sorted(words, key=lambda word: (word.bbox.y0 + word.bbox.y1) / 2)
    rows: list[list[Word]] = []
    for word in ordered:
        center = (word.bbox.y0 + word.bbox.y1) / 2
        if rows:
            last = rows[-1]
            last_center = float(np.mean([(w.bbox.y0 + w.bbox.y1) / 2 for w in last]))
            if abs(center - last_center) <= tolerance:
                last.append(word)
                continue
        rows.append([word])
    for row in rows:
        row.sort(key=lambda word: word.bbox.x0)
    return rows


def assign_words(boxes: list[BBox], words: list[Word]) -> list[list[Word]]:
    """把每个词分配给"包含它中心点"的块；不在任何块里的词给最近的块。

    Args:
        boxes: 内容块。
        words: 词列表。

    Returns:
        与 ``boxes`` 等长的列表，每项是该块的词。

    Raises:
        ValueError: 传入空块列表（调用方应先处理空页）。
    """
    if not boxes:
        msg = "assign_words 需要至少一个内容块"
        raise ValueError(msg)

    assigned: list[list[Word]] = [[] for _ in boxes]
    for word in words:
        cx = (word.bbox.x0 + word.bbox.x1) // 2
        cy = (word.bbox.y0 + word.bbox.y1) // 2
        index = _owner_index(boxes, cx, cy)
        assigned[index].append(word)
    return assigned


def _owner_index(boxes: list[BBox], x: int, y: int) -> int:
    """找到包含点 (x, y) 的块；否则返回中心最近的块。

    Args:
        boxes: 内容块。
        x: 点 x。
        y: 点 y。

    Returns:
        块下标。
    """
    for index, box in enumerate(boxes):
        if box.x0 <= x < box.x1 and box.y0 <= y < box.y1:
            return index
    distances = [
        ((box.x0 + box.x1) / 2 - x) ** 2 + ((box.y0 + box.y1) / 2 - y) ** 2 for box in boxes
    ]
    return int(np.argmin(distances))


def merge_overlapping(boxes: list[BBox], iou_threshold: float = 0.3) -> list[BBox]:
    """合并高度重叠的块（XY-cut 偶尔会把同一段切成两块）。

    Args:
        boxes: 原始块。
        iou_threshold: 超过该 IoU 即合并。

    Returns:
        合并后的块，保持原有大致顺序。
    """
    merged: list[BBox] = []
    for box in boxes:
        target = _first_overlapping(merged, box, iou_threshold)
        if target is None:
            merged.append(box)
        else:
            merged[target] = merged[target].merge(box)
    return merged


def _first_overlapping(boxes: list[BBox], box: BBox, threshold: float) -> int | None:
    """返回第一个与 ``box`` 足够重叠的下标。

    Args:
        boxes: 已收集的块。
        box: 待比较的块。
        threshold: IoU 阈值。

    Returns:
        下标或 None。
    """
    for index, candidate in enumerate(boxes):
        if candidate.iou(box) >= threshold:
            return index
    return None


def _drop_tiny(boxes: list[BBox], min_area: int = 900) -> list[BBox]:
    """丢弃面积过小的块（噪点、单字符标点）。

    Args:
        boxes: 块列表。
        min_area: 最小面积（像素）。

    Returns:
        过滤后的块。
    """
    return [box for box in boxes if box.area >= min_area]


def _mean_confidence(words: list[Word]) -> float:
    """区域平均 OCR 置信度；无词时为 0。

    Args:
        words: 区域内的词。

    Returns:
        0~1 的均值。
    """
    if not words:
        return 0.0
    return float(sum(word.confidence for word in words) / len(words))


def reading_order(regions: list[Region]) -> list[Region]:
    """按阅读顺序排序：先按纵向分带，再按横向。

    Args:
        regions: 区域列表。

    Returns:
        排序后的新列表。
    """
    band = defaultdict(list)
    for region in regions:
        band[region.bbox.y0 // 40].append(region)
    ordered: list[Region] = []
    for key in sorted(band):
        ordered.extend(sorted(band[key], key=lambda region: region.bbox.x0))
    return ordered
