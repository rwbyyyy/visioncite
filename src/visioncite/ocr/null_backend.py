"""空 OCR 后端：没有 OCR 依赖时的降级实现。

它的价值在于"让系统在任何环境都能启动"：
- CI 环境不装 OCR，也能跑通除文字识别外的全部流程；
- 用户只想用 PDF 文本层时，可以显式设置 ``VISIONCITE_OCR=off``。

面试点：降级不是"功能缺失"，而是**明确的、可观测的能力开关**，
/health 会报告当前用的是哪个后端。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from visioncite.types import Word


class NullOcrBackend:
    """什么都不做的后端。"""

    name = "null"

    def available(self) -> bool:
        """始终可用（空实现本身就是一种可用状态）。"""
        return True

    def recognize(self, image: NDArray[np.uint8]) -> list[Word]:
        """返回空列表。

        Args:
            image: 未使用。

        Returns:
            空列表。
        """
        return []
