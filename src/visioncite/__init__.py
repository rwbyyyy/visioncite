"""VisionCite（证源）：多模态文档问答与证据溯源。

顶层只导出最常用的公共对象，保持包的 API 面很小：

    from visioncite import Settings, DocumentStore, answer_question

设计原则：外部只依赖"稳定契约"（本文件 + types.py），
内部模块可以自由重构而不影响调用方。
"""

from visioncite.config import Settings
from visioncite.errors import VisionCiteError
from visioncite.types import (
    AnswerResult,
    BBox,
    Chunk,
    Citation,
    DocumentMeta,
    Page,
    Region,
    Word,
)

__version__ = "0.2.0"

__all__ = [
    "AnswerResult",
    "BBox",
    "Chunk",
    "Citation",
    "DocumentMeta",
    "Page",
    "Region",
    "Settings",
    "VisionCiteError",
    "Word",
    "__version__",
]
