"""集中配置：所有可调参数只有一个来源。

为什么不用散落的常量？
    面试常问"这个 0.35 是什么"——把它放进 ``Settings`` 后，
    它有了名字、默认值、可被环境变量覆盖，还能被 /health 打印出来。
    调整参数只改一处，评测与线上不会出现"两套数值"。

环境变量命名统一用 ``VISIONCITE_`` 前缀，便于容器化部署。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OcrMode = Literal["auto", "off"]
VisualMode = Literal["auto", "off"]


def _env_str(name: str, default: str) -> str:
    """读取字符串环境变量，空值回退默认。"""
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，非法值回退默认（不让配置错误炸掉服务）。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量，非法值回退默认。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """运行时配置。

    Attributes:
        cache_dir: 缓存根目录（页面图、模型、索引都放这里）。
        render_dpi: PDF 渲染分辨率；越高 OCR 越准，越慢越占内存。
        max_image_side: 图片最长边上限，超过先缩放，防止大图拖垮 OCR。
        max_upload_mb: 上传大小上限。
        bm25_k1: BM25 词频饱和参数，经典取值 1.2~2.0。
        bm25_b: BM25 长度归一强度，0 表示不归一，1 表示完全归一。
        top_k: 每次检索保留的证据块数。
        rrf_k: RRF 平滑常数，论文推荐 60。
        mmr_lambda: MMR 的相关性/多样性权衡，1.0 等于纯相关性。
        abstain_threshold: 低于此置信度则弃答。
        min_citation_score: 单条证据的最低分，低于它的引用不展示。
        ocr_mode: ``auto`` 尝试加载 OCR，``off`` 强制关闭。
        visual_mode: ``auto`` 尝试加载 CLIP，``off`` 强制关闭。
        clip_model: CLIP 模型 id（本地目录或 HuggingFace 仓库名）。
        hf_endpoint: HuggingFace 镜像地址，国内网络建议用 hf-mirror。
        vlm_base_url: 可选 VLM 的 OpenAI 兼容地址；为空表示不启用。
        vlm_model: 可选 VLM 的模型名。
    """

    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "visioncite")
    render_dpi: int = 144
    max_image_side: int = 2400
    max_upload_mb: int = 25

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k: int = 8
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    abstain_threshold: float = 0.35
    min_citation_score: float = 0.02

    ocr_mode: OcrMode = "auto"
    visual_mode: VisualMode = "auto"
    clip_model: str = "openai/clip-vit-base-patch32"
    hf_endpoint: str = "https://hf-mirror.com"

    vlm_base_url: str = ""
    vlm_model: str = ""

    llm_base_url: str = ""
    llm_model: str = ""
    agent_max_steps: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量构造配置。

        Returns:
            配置对象；任何非法取值都会安全回退到默认值。
        """
        return cls(
            cache_dir=Path(_env_str("VISIONCITE_CACHE", str(Path.home() / ".cache" / "visioncite"))),
            render_dpi=_env_int("VISIONCITE_RENDER_DPI", 144),
            max_image_side=_env_int("VISIONCITE_MAX_IMAGE_SIDE", 2400),
            max_upload_mb=_env_int("VISIONCITE_MAX_UPLOAD_MB", 25),
            bm25_k1=_env_float("VISIONCITE_BM25_K1", 1.5),
            bm25_b=_env_float("VISIONCITE_BM25_B", 0.75),
            top_k=_env_int("VISIONCITE_TOP_K", 8),
            rrf_k=_env_int("VISIONCITE_RRF_K", 60),
            mmr_lambda=_env_float("VISIONCITE_MMR_LAMBDA", 0.7),
            abstain_threshold=_env_float("VISIONCITE_ABSTAIN_THRESHOLD", 0.35),
            min_citation_score=_env_float("VISIONCITE_MIN_CITATION_SCORE", 0.02),
            ocr_mode="off" if _env_str("VISIONCITE_OCR", "auto") == "off" else "auto",
            visual_mode="off" if _env_str("VISIONCITE_VISUAL", "auto") == "off" else "auto",
            clip_model=_env_str("VISIONCITE_CLIP_MODEL", "openai/clip-vit-base-patch32"),
            hf_endpoint=_env_str("VISIONCITE_HF_ENDPOINT", "https://hf-mirror.com"),
            vlm_base_url=_env_str("VISIONCITE_VLM_BASE_URL", ""),
            vlm_model=_env_str("VISIONCITE_VLM_MODEL", ""),
            llm_base_url=_env_str("VISIONCITE_LLM_BASE_URL", ""),
            llm_model=_env_str("VISIONCITE_LLM_MODEL", ""),
            agent_max_steps=_env_int("VISIONCITE_AGENT_MAX_STEPS", 4),
        )

    @property
    def model_dir(self) -> Path:
        """模型权重目录（与索引缓存放一起，便于整体迁移）。"""
        return self.cache_dir / "models"

    @property
    def work_dir(self) -> Path:
        """运行时工作目录：上传原图、渲染页图、索引文件。"""
        return self.cache_dir / "work"

    @property
    def vlm_enabled(self) -> bool:
        """是否配置了可选 VLM。未配置时走纯离线的 extractive 路径。"""
        return bool(self.vlm_base_url and self.vlm_model)

    @property
    def llm_enabled(self) -> bool:
        """是否配置了 Agent 用的 LLM。未配置时 CLI 退回 fake provider。"""
        return bool(self.llm_base_url and self.llm_model)

    def ensure_dirs(self) -> None:
        """创建运行所需目录（幂等，可重复调用）。"""
        for path in (self.cache_dir, self.model_dir, self.work_dir):
            path.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        """给 /health 用的可观测摘要（不含任何密钥）。"""
        return {
            "cache_dir": str(self.cache_dir),
            "ocr_mode": self.ocr_mode,
            "visual_mode": self.visual_mode,
            "clip_model": self.clip_model,
            "vlm_enabled": self.vlm_enabled,
            "abstain_threshold": self.abstain_threshold,
            "top_k": self.top_k,
        }
