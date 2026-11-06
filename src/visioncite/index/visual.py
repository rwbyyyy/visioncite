"""CLIP 视觉编码器：把图片和文字映射到同一个向量空间。

CLIP 的核心思想（面试必问）：
    用 4 亿对（图片, 网络文本）做对比学习。一个 batch 有 N 对，
    对角线上是正确配对，其余 N²-N 个是负例；
    损失是对称的 InfoNCE：
        L = -1/N Σ_i log( exp(sim(i,i)/τ) / Σ_j exp(sim(i,j)/τ) )
    训练完，图片和"描述它的文字"在向量空间里靠近，
    于是可以用文字直接检索图片（zero-shot），不需要为每类图片训练分类头。

在我们项目里的作用：
    给每个版面区域（图、表、公式块）算向量，查询时与文本向量做余弦相似度，
    和 BM25 的名次做 RRF 融合。

已知失效（要主动说，而不是被问出来）：
    CLIP 的文本编码器在"密集小字、表格数字"上很弱，
    它擅长"这张图大概是什么"，不擅长"这张发票的金额是多少"。
    因此它只作为**视觉召回**的一路，不做最终答案依据。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from visioncite.config import Settings
from visioncite.errors import VisualUnavailableError
from visioncite.logging import get_logger

logger = get_logger("index.visual")


class ClipVisualEncoder:
    """CLIP 封装：懒加载 + 失败即降级。"""

    name = "clip"

    def __init__(self, model_id: str, cache_dir: Path, batch_size: int = 8) -> None:
        """初始化（不加载模型）。

        Args:
            model_id: HuggingFace 仓库名或本地目录。
            cache_dir: 权重缓存目录。
            batch_size: 推理批大小，CPU 上 8 比较稳。
        """
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model: object | None = None
        self._processor: object | None = None
        self._load_failed = False

    def available(self) -> bool:
        """只检查依赖是否安装，不触发下载。"""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return not self._load_failed

    def load(self) -> bool:
        """加载模型与预处理器（首次会下载权重）。

        Returns:
            True 表示可用。
        """
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info("加载 CLIP 模型 %s（首次会下载权重）", self._model_id)
            self._model = CLIPModel.from_pretrained(self._model_id, cache_dir=str(self._cache_dir))
            self._processor = CLIPProcessor.from_pretrained(
                self._model_id, cache_dir=str(self._cache_dir)
            )
            self._model.eval()  # type: ignore[union-attr]
            logger.info("CLIP 就绪")
            return True
        except Exception:  # noqa: BLE001 - 网络/权重问题都按降级处理
            self._load_failed = True
            logger.exception("CLIP 加载失败，视觉检索关闭")
            return False

    def encode_images(self, images: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """编码一组图片。

        Args:
            images: RGB 数组列表。

        Returns:
            ``(N, D)`` 的 L2 归一化向量。

        Raises:
            VisualUnavailableError: 模型不可用。
        """
        if not self.load():
            msg = "CLIP 不可用"
            raise VisualUnavailableError(msg)

        import torch
        from PIL import Image

        vectors: list[NDArray[np.float32]] = []
        with torch.no_grad():
            for start in range(0, len(images), self._batch_size):
                batch = [
                    Image.fromarray(image) for image in images[start : start + self._batch_size]
                ]
                inputs = self._processor(images=batch, return_tensors="pt")  # type: ignore[operator]
                vectors.append(_to_numpy(self._image_features(inputs["pixel_values"])))
        return _l2_normalize(np.concatenate(vectors, axis=0))

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """编码一组文本。

        Args:
            texts: 文本列表。

        Returns:
            ``(N, D)`` 的 L2 归一化向量。

        Raises:
            VisualUnavailableError: 模型不可用。
        """
        if not self.load():
            msg = "CLIP 不可用"
            raise VisualUnavailableError(msg)

        import torch

        vectors: list[NDArray[np.float32]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                inputs = self._processor(  # type: ignore[operator]
                    text=batch, padding=True, truncation=True, return_tensors="pt"
                )
                vectors.append(_to_numpy(self._text_features(inputs)))
        return _l2_normalize(np.concatenate(vectors, axis=0))

    def _image_features(self, pixel_values: object) -> object:
        """取图像向量。

        transformers 5.x 的 ``CLIPModel.forward`` 要求同时给文本输入，
        因此这里直接调用视觉塔 + 投影层，跨版本更稳。

        Args:
            pixel_values: 预处理后的像素张量。

        Returns:
            图像向量张量。
        """
        model = self._model
        if hasattr(model, "vision_model") and hasattr(model, "visual_projection"):
            pooled = model.vision_model(pixel_values=pixel_values).pooler_output  # type: ignore[union-attr]
            return model.visual_projection(pooled)  # type: ignore[union-attr]
        return _pick_embeds(model.get_image_features(pixel_values=pixel_values), "image")  # type: ignore[union-attr]

    def _text_features(self, inputs: dict) -> object:
        """取文本向量。

        Args:
            inputs: 预处理后的 ``input_ids`` / ``attention_mask``。

        Returns:
            文本向量张量。
        """
        model = self._model
        if hasattr(model, "text_model") and hasattr(model, "text_projection"):
            pooled = model.text_model(  # type: ignore[union-attr]
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            ).pooler_output
            return model.text_projection(pooled)  # type: ignore[union-attr]
        return _pick_embeds(model.get_text_features(**inputs), "text")  # type: ignore[union-attr]


def _pick_embeds(output: object, kind: str) -> object:
    """从模型输出里取向量，兼容 transformers 不同版本的字段名。

    Args:
        output: 模型返回对象。
        kind: ``"image"`` 或 ``"text"``。

    Returns:
        张量。
    """
    for name in (f"{kind}_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    return output


def _to_numpy(tensor: object) -> NDArray[np.float32]:
    """把 torch 张量转 numpy。

    Args:
        tensor: 张量。

    Returns:
        float32 数组。
    """
    return tensor.cpu().numpy().astype(np.float32)  # type: ignore[attr-defined]


class NullVisualEncoder:
    """视觉编码关闭时的占位实现，保证调用方代码不需要分支。"""

    name = "null"

    def available(self) -> bool:
        """始终不可用。"""
        return False

    def load(self) -> bool:
        """始终失败。"""
        return False

    def encode_images(self, images: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """抛异常，提示调用方走降级分支。"""
        msg = "视觉编码器已关闭"
        raise VisualUnavailableError(msg)

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """抛异常，提示调用方走降级分支。"""
        msg = "视觉编码器已关闭"
        raise VisualUnavailableError(msg)


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """按行做 L2 归一化，使余弦相似度等于点积。

    Args:
        matrix: ``(N, D)`` 矩阵。

    Returns:
        归一化后的矩阵；零向量保持为零。
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def create_visual(settings: Settings) -> ClipVisualEncoder | NullVisualEncoder:
    """按配置创建视觉编码器。

    Args:
        settings: 运行配置。

    Returns:
        CLIP 编码器或空实现。
    """
    if settings.visual_mode == "off":
        logger.info("视觉检索已按配置关闭")
        return NullVisualEncoder()
    encoder = ClipVisualEncoder(settings.clip_model, settings.model_dir)
    if encoder.available():
        return encoder
    logger.warning("未安装 torch/transformers，视觉检索关闭")
    return NullVisualEncoder()
