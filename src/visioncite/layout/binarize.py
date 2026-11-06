"""图像二值化：把"哪里有墨"变成布尔矩阵。

基础概念（面试常问）：
- 灰度化：``Y = 0.299R + 0.587G + 0.114B``，这是 ITU-R BT.601 的亮度公式，
  权重不同是因为人眼对绿色最敏感。
- Otsu 阈值：在所有 256 个候选阈值里，选让"前景/背景两类之间的方差"最大的那个。
  它是无参数的，对扫描件很稳；缺点是对光照不均（拍照阴影）不鲁棒，
  那种场景应该用自适应阈值（见 docs/EXERCISES.md 练习 1）。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from visioncite.types import BBox


def to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """RGB 转灰度。

    Args:
        image: ``(H, W, 3)`` 的 RGB 数组。

    Returns:
        ``(H, W)`` 的灰度数组。
    """
    if image.ndim == 2:  # 已经是灰度图
        return image
    # 用浮点乘加再转回 uint8：整数运算容易溢出，这里图不大，性能可接受
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.tensordot(image.astype(np.float32), weights, axes=([2], [0]))
    return np.clip(gray, 0, 255).astype(np.uint8)


def otsu_threshold(gray: NDArray[np.uint8]) -> int:
    """用 Otsu 法求全局阈值。

    数学推导：
        设阈值为 t，像素被分成 <=t（背景）和 >t（前景）两类，比例为 w0、w1，
        均值为 mu0、mu1。类间方差
            sigma_b^2(t) = w0 * w1 * (mu0 - mu1)^2
        使 sigma_b^2 最大的 t 就是 Otsu 阈值。
        等价写法（用累积和算，避免循环）：
            sigma_b^2 = (mu_T * w0 - mu_0)^2 / (w0 * (1 - w0))

    Args:
        gray: 灰度图。

    Returns:
        0~255 的阈值（整数）。
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    prob = hist / total
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(prob)  # w0(t)
    mu = np.cumsum(prob * levels)  # mu_0(t)
    mu_total = mu[-1]

    denominator = omega * (1.0 - omega)
    # 分母为 0 的位置（全黑/全白阈值）没有意义，置 0 避免除零告警
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(denominator > 1e-12, (mu_total * omega - mu) ** 2 / denominator, 0.0)
    return int(np.argmax(variance))


def ink_mask(gray: NDArray[np.uint8], threshold: int | None = None) -> NDArray[np.bool_]:
    """生成"墨迹"布尔矩阵：True 表示该像素是深色内容。

    注意用 ``<=`` 而不是 ``<``：纯黑白扫描件上 Otsu 的最优阈值常常就是 0，
    用 ``<`` 会把所有黑像素漏掉。

    Args:
        gray: 灰度图。
        threshold: 指定阈值；None 表示用 Otsu 自动求。

    Returns:
        与灰度图同形状的布尔数组。
    """
    cut = otsu_threshold(gray) if threshold is None else threshold
    return gray <= cut


def ink_density(mask: NDArray[np.bool_], bbox: BBox) -> float:
    """区域内的墨迹占比。

    Args:
        mask: 布尔墨迹矩阵。
        bbox: 区域。

    Returns:
        0~1 的比例；区域越界时裁剪后再算，空区域返回 0。
    """
    height, width = mask.shape[:2]
    x0 = max(0, min(width, bbox.x0))
    x1 = max(0, min(width, bbox.x1))
    y0 = max(0, min(height, bbox.y0))
    y1 = max(0, min(height, bbox.y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = mask[y0:y1, x0:x1]
    return float(patch.mean())
