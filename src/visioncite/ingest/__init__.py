"""文档接入层：把 PDF / 图片变成统一的 Page 数组。"""

from visioncite.ingest.image import crop_image, hash_distance, load_image, perceptual_hash
from visioncite.ingest.loader import detect_kind, load_pages

__all__ = [
    "crop_image",
    "detect_kind",
    "hash_distance",
    "load_image",
    "load_pages",
    "perceptual_hash",
]
