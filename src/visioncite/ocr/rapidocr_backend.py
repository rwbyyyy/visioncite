"""RapidOCR 后端：离线、CPU、中英文。

选它的原因：
1. 基于 ONNXRuntime，不依赖 GPU，安装即用（wheel 内置模型）；
2. 中英文混排识别质量在同量级里够用；
3. 完全离线，适合内网/隐私场景，也方便面试时现场演示。

调用约定（1.3+）：
    engine = RapidOCR()
    result, elapsed = engine(image)     # image 可以是 ndarray
    result = [[四点坐标, 文本, 置信度], ...]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from visioncite.logging import get_logger
from visioncite.ocr.base import filter_confidence, polygon_to_bbox
from visioncite.types import Word

logger = get_logger("ocr.rapidocr")


class RapidOcrBackend:
    """RapidOCR 封装：懒加载引擎，避免 import 时就下载/初始化。"""

    name = "rapidocr"

    def __init__(self, min_confidence: float = 0.5) -> None:
        """初始化。

        Args:
            min_confidence: 低于该置信度的词会被丢弃。
        """
        self._engine: object | None = None
        self._load_failed = False
        self._min_confidence = min_confidence

    def available(self) -> bool:
        """检查 rapidocr 是否已安装。"""
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError:
            return False
        return not self._load_failed

    def recognize(self, image: NDArray[np.uint8]) -> list[Word]:
        """识别整页。

        Args:
            image: ``(H, W, 3)`` RGB 图。

        Returns:
            词列表；识别失败时返回空列表（不抛异常，保证服务可用）。
        """
        engine = self._get_engine()
        if engine is None:
            return []
        try:
            raw, _elapsed = engine(image)  # type: ignore[operator]
        except Exception:  # noqa: BLE001 - OCR 失败只降级，不阻断整条链路
            logger.exception("OCR 推理失败，本页按无文本处理")
            return []
        return self._to_words(raw)

    def _get_engine(self) -> object | None:
        """懒加载引擎。

        Returns:
            引擎对象；加载失败返回 None。
        """
        if self._engine is not None or self._load_failed:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
            logger.info("RapidOCR 已加载")
        except Exception:  # noqa: BLE001
            self._load_failed = True
            logger.exception("RapidOCR 加载失败，降级为无 OCR")
        return self._engine

    def _to_words(self, raw: object) -> list[Word]:
        """把原始输出转成 Word 列表。

        Args:
            raw: rapidocr 的返回值，通常为 ``[[box, text, score], ...]`` 或 None。

        Returns:
            过滤后的词列表。
        """
        if not raw:
            return []
        words: list[Word] = []
        for item in raw:  # type: ignore[union-attr]
            try:
                box, text, score = item[0], str(item[1]), float(item[2])
            except (TypeError, IndexError, ValueError):
                continue
            if not text.strip():
                continue
            words.append(Word(text=text.strip(), bbox=polygon_to_bbox(box), confidence=score))
        return filter_confidence(words, self._min_confidence)


def create_rapidocr(min_confidence: float = 0.5) -> RapidOcrBackend:
    """工厂函数：方便测试时替换。

    Args:
        min_confidence: 置信度阈值。

    Returns:
        后端实例。
    """
    return RapidOcrBackend(min_confidence=min_confidence)
