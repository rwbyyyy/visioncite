"""全项目共享的数据契约（dataclass）。

为什么单独放一个模块？
    如果每个子模块各写一套 dict，接口就会在重构时悄悄错位。
    这里把"页面 / 区域 / 词 / 引用 / 答案"定义成不可变或半不可变对象，
    任何模块只能按这份契约读写，类型检查器（mypy/IDE）也能顺着找到定义。

坐标约定（很重要，后面所有 bbox 都遵守）：
    原点在左上角，x 向右，y 向下，单位是像素；
    ``bbox = (x0, y0, x1, y1)`` 表示**左闭右开**区间，x1/y1 不包含在内。
    左闭右开的好处是面积可以直接写 ``(x1-x0)*(y1-y0)``，拼接相邻块也不会重复计像素。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# 区域类型：先粗分四类，够用且容易解释；扩类型时只改这里 + 分类函数
RegionKind = Literal["text", "table", "figure", "unknown"]


@dataclass(frozen=True)
class BBox:
    """轴对齐矩形区域。

    Attributes:
        x0: 左边界（含）。
        y0: 上边界（含）。
        x1: 右边界（不含）。
        y1: 下边界（不含）。
    """

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        """规范化坐标，保证 x0<=x1、y0<=y1。

        frozen=True 的 dataclass 不能直接赋值，需要绕过 setattr；
        这样上游哪怕传反了也不会在 IoU 里算出负数面积。
        """
        if self.x0 > self.x1:
            low, high = self.x1, self.x0
            object.__setattr__(self, "x0", low)
            object.__setattr__(self, "x1", high)
        if self.y0 > self.y1:
            low, high = self.y1, self.y0
            object.__setattr__(self, "y0", low)
            object.__setattr__(self, "y1", high)

    @property
    def width(self) -> int:
        """矩形宽度（像素）。"""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """矩形高度（像素）。"""
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        """矩形面积（像素）。"""
        return self.width * self.height

    def intersection_area(self, other: BBox) -> int:
        """与另一个矩形的交集面积。

        Args:
            other: 另一个矩形。

        Returns:
            交集面积，不相交时为 0。
        """
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        if dx <= 0 or dy <= 0:
            return 0
        return dx * dy

    def iou(self, other: BBox) -> float:
        """交并比（Intersection over Union）。

        IoU 是目标检测/证据匹配的标准度量：交集 / 并集。
        评测里用 ``IoU >= 0.5`` 判定"引用命中"，与 COCO 的 AP50 同一约定。

        Args:
            other: 另一个矩形。

        Returns:
            0~1 之间的实数；两个矩形都为空时返回 0。
        """
        inter = self.intersection_area(other)
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def merge(self, other: BBox) -> BBox:
        """返回包裹两个矩形的最小外接矩形。"""
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def pad(self, px: int, bounds: tuple[int, int] | None = None) -> BBox:
        """向外扩张若干像素，并可被页面边界裁剪。

        Args:
            px: 向外扩张的像素数。
            bounds: ``(宽, 高)``；给出时把结果裁剪到页面内，避免越界。

        Returns:
            新的 BBox。
        """
        box = BBox(self.x0 - px, self.y0 - px, self.x1 + px, self.y1 + px)
        if bounds is None:
            return box
        width, height = bounds
        return BBox(
            max(0, box.x0),
            max(0, box.y0),
            min(width, box.x1),
            min(height, box.y1),
        )

    def to_list(self) -> list[int]:
        """转成 ``[x0, y0, x1, y1]``，便于写 JSON。"""
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values: list[int] | tuple[int, ...]) -> BBox:
        """从 ``[x0, y0, x1, y1]`` 构造。

        Args:
            values: 长度为 4 的序列。

        Returns:
            新的 BBox。

        Raises:
            ValueError: 长度不是 4。
        """
        if len(values) != 4:
            msg = f"bbox 需要 4 个数字，收到 {len(values)} 个"
            raise ValueError(msg)
        return cls(*(int(v) for v in values))


@dataclass(frozen=True)
class Word:
    """一个 OCR 词/字块，带位置与置信度。

    Attributes:
        text: 识别文本。
        bbox: 在页面坐标系中的位置。
        confidence: 0~1，OCR 给出的置信度；人工构造的数据默认 1.0。
    """

    text: str
    bbox: BBox
    confidence: float = 1.0


@dataclass
class Region:
    """版面区域：一段正文、一个表格或一张图。

    区域是"证据"的最小单位：答案引用的就是 ``region_id``。

    Attributes:
        region_id: 页内唯一 id，格式 ``p{页码}-r{序号}``。
        page: 页码，从 0 开始。
        bbox: 区域位置。
        kind: 区域类型。
        text: 区域内文本（表格会把单元格按行拼成 ``a | b``）。
        words: 区域内 OCR 词，供进一步切句。
        confidence: 该区域文本的平均 OCR 置信度。
    """

    region_id: str
    page: int
    bbox: BBox
    kind: RegionKind = "text"
    text: str = ""
    words: list[Word] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Page:
    """一页文档（PDF 的一页 / 一张图片）。

    Attributes:
        index: 页码，从 0 开始。
        width: 页面像素宽。
        height: 页面像素高。
        image: RGB 数组，形状 ``(H, W, 3)``，dtype=uint8。
        regions: 版面区域列表。
        text_layer: PDF 自带的文本层（可能为空，扫描件就是空）。
    """

    index: int
    width: int
    height: int
    image: NDArray[np.uint8]
    regions: list[Region] = field(default_factory=list)
    text_layer: str = ""


@dataclass(frozen=True)
class Chunk:
    """检索单元：区域文本 + 出处。

    与 Region 的区别：Chunk 会被切成更小的检索粒度（长区域按句切），
    但保留 ``region_id`` 指回原区域，回答时才能给出可视化证据。
    """

    chunk_id: str
    doc_id: str
    page: int
    region_id: str
    bbox: BBox
    kind: RegionKind
    text: str


@dataclass(frozen=True)
class Citation:
    """答案引用的一条证据。

    Attributes:
        chunk_id: 命中的检索块。
        region_id: 对应版面区域。
        page: 页码。
        bbox: 证据位置，前端据此在页面图上画框。
        snippet: 证据原文片段。
        score: 融合检索分数（越大越相关）。
    """

    chunk_id: str
    region_id: str
    page: int
    bbox: BBox
    snippet: str
    score: float = 0.0


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的完整结果。

    Attributes:
        answer: 答案文本；弃答时为空串。
        citations: 证据列表，按相关度排序。
        abstained: 是否弃答（证据不足或校验不通过）。
        confidence: 0~1 的置信度，用于校准与评测。
        latency_ms: 端到端耗时。
        backend: ``extractive`` 或 ``vlm``。
        reasons: 弃答/降级原因，便于排查，不展示给终端用户。
    """

    answer: str
    citations: tuple[Citation, ...] = ()
    abstained: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0
    backend: str = "extractive"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentMeta:
    """文档元信息，API 返回与索引落盘都要用。

    Attributes:
        doc_id: 文档 id（内容指纹），同一文件重复上传会命中去重。
        filename: 原始文件名。
        page_count: 页数。
        size_bytes: 原始字节数。
    """

    doc_id: str
    filename: str
    page_count: int
    size_bytes: int = 0
