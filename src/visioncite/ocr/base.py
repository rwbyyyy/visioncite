"""OCR 后端的公共契约。

为什么要抽象成 Protocol？
    生产上 OCR 可能换成 PaddleOCR、云服务或自研模型；
    上层（版面、检索）不应该知道具体实现。
    这里定义"能识别一张图并返回词 + 位置 + 置信度"这一条最小接口，
    换后端只需新增一个类，不改调用方（依赖倒置原则）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from visioncite.types import BBox, Word


def polygon_to_bbox(points: list[list[float]] | NDArray[np.floating]) -> BBox:
    """把四边形（OCR 常见输出）转成轴对齐矩形。

    Args:
        points: 形如 ``[[x, y], ...]`` 的顶点序列。

    Returns:
        包围这些点的矩形。
    """
    coords = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return BBox(
        int(np.floor(coords[:, 0].min())),
        int(np.floor(coords[:, 1].min())),
        int(np.ceil(coords[:, 0].max())),
        int(np.ceil(coords[:, 1].max())),
    )


def filter_confidence(words: list[Word], min_confidence: float) -> list[Word]:
    """过滤掉低置信度的词。

    低置信度往往来自噪点、印章纹理、图片里的花纹文字。
    保留它们会污染检索（"看起来像字"的噪声进入 BM25 语料）。

    Args:
        words: OCR 结果。
        min_confidence: 最低置信度（0~1）。

    Returns:
        过滤后的词列表。
    """
    return [word for word in words if word.confidence >= min_confidence]


@runtime_checkable
class OcrBackend(Protocol):
    """OCR 后端接口（结构化子类型，不需要继承）。"""

    name: str

    def available(self) -> bool:
        """后端是否可用（依赖已安装、模型已就绪）。"""
        ...

    def recognize(self, image: NDArray[np.uint8]) -> list[Word]:
        """识别一张 RGB 图。"""
        ...
