"""OCR 层：接口 + 两个实现（RapidOCR / 空实现）。"""

from visioncite.ocr.base import OcrBackend, filter_confidence, polygon_to_bbox
from visioncite.ocr.factory import create_ocr
from visioncite.ocr.null_backend import NullOcrBackend
from visioncite.ocr.rapidocr_backend import RapidOcrBackend

__all__ = [
    "NullOcrBackend",
    "OcrBackend",
    "RapidOcrBackend",
    "create_ocr",
    "filter_confidence",
    "polygon_to_bbox",
]
