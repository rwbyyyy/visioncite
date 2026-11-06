"""OCR 工厂：根据配置选择后端。

调用方只拿 ``OcrBackend`` 接口，不关心具体实现：
    backend = create_ocr(settings)
    words = backend.recognize(image)
"""

from __future__ import annotations

from visioncite.config import Settings
from visioncite.logging import get_logger
from visioncite.ocr.base import OcrBackend
from visioncite.ocr.null_backend import NullOcrBackend
from visioncite.ocr.rapidocr_backend import RapidOcrBackend

logger = get_logger("ocr.factory")


def create_ocr(settings: Settings) -> OcrBackend:
    """按配置创建 OCR 后端。

    顺序：显式关闭 → Null；否则尝试 RapidOCR，导入失败 → Null。

    Args:
        settings: 运行配置。

    Returns:
        可用后端。
    """
    if settings.ocr_mode == "off":
        logger.info("OCR 已按配置关闭")
        return NullOcrBackend()

    backend = RapidOcrBackend()
    if backend.available():
        return backend
    logger.warning("未安装 rapidocr-onnxruntime，降级为无 OCR")
    return NullOcrBackend()
