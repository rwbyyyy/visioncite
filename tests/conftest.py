"""测试公共夹具。

原则：
1. 不联网、不下载模型、不依赖本机 OCR —— 全部用假实现或合成图；
2. 每个夹具只做一件事，测试出错时能快速定位；
3. 所有测试产物写到 tmp_path，不污染仓库。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visioncite.config import Settings  # noqa: E402
from visioncite.ocr.base import OcrBackend  # noqa: E402
from visioncite.types import BBox, Word  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """把缓存目录指到临时目录的配置。"""
    config = Settings(cache_dir=tmp_path / "cache")
    config.ensure_dirs()
    return config


class FakeOcr(OcrBackend):
    """返回预置词的假 OCR：测试版面/检索/答案时不依赖真实模型。"""

    name = "fake"

    def __init__(self, words: list[Word]) -> None:
        self.words = words

    def available(self) -> bool:
        return True

    def recognize(self, image: np.ndarray) -> list[Word]:
        return list(self.words)


def make_document_image(tmp_path: Path) -> tuple[Path, list[Word]]:
    """画一张"两行文字"的图，并按坐标给出对应词。

    Args:
        tmp_path: 临时目录。

    Returns:
        ``(图片路径, 词列表)``。
    """
    image = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(image)
    # 用矩形模拟文字行（不依赖字体文件，保证跨平台可跑）
    draw.rectangle([60, 80, 520, 110], fill="black")
    draw.rectangle([60, 200, 420, 230], fill="black")
    path = tmp_path / "page.png"
    image.save(path)
    words = [
        Word("合计金额：1286.40元", BBox(60, 80, 520, 110)),
        Word("开票日期：2026-10-25", BBox(60, 200, 420, 230)),
    ]
    return path, words


@pytest.fixture
def document_path(tmp_path: Path) -> tuple[Path, list[Word]]:
    """提供一张合成页面与对应词。"""
    return make_document_image(tmp_path)


@pytest.fixture
def fake_ocr(document_path: tuple[Path, list[Word]]) -> FakeOcr:
    """与合成页面对应的假 OCR。"""
    return FakeOcr(document_path[1])
