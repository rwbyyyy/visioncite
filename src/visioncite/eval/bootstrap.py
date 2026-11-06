"""统计工具：置信区间与配对检验。

为什么评测要带置信区间？
    60 条样本算出来的 F1 是 0.81，换一批样本可能就是 0.72。
    只报一个点估计，等于假装样本无限大。bootstrap 不假设分布，
    实现只有几行，面试时能现场手推：有放回重采样 B 次，取分位点。

配对检验用于消融：
    "加了视觉检索后 F1 提升 0.04" —— 这个提升是真的，还是噪声？
    对同一批样本做配对 bootstrap（每次对两条方法的差值分布重采样），
    看 95% 区间是否包含 0。
"""

from __future__ import annotations

import random


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """均值的 bootstrap 置信区间。

    Args:
        values: 样本。
        n_boot: 重采样次数。
        alpha: 显著性水平，0.05 对应 95% 区间。
        seed: 随机种子，保证可复现。

    Returns:
        ``(下界, 上界)``；样本为空时返回 (0, 0)。
    """
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    lower = means[int((alpha / 2) * n_boot)]
    upper = means[int((1 - alpha / 2) * n_boot) - 1]
    return lower, upper


def paired_bootstrap_ci(
    a: list[float],
    b: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """两条方法差值的配对 bootstrap 区间。

    Args:
        a: 方法 A 的逐样本得分。
        b: 方法 B 的逐样本得分（与 A 对齐）。
        n_boot: 重采样次数。
        alpha: 显著性水平。
        seed: 随机种子。

    Returns:
        ``(下界, 上界)``；区间不含 0 时通常认为差异显著。

    Raises:
        ValueError: 两个列表长度不一致。
    """
    if len(a) != len(b):
        msg = "配对检验要求两组样本一一对应"
        raise ValueError(msg)
    if not a:
        return 0.0, 0.0
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    return bootstrap_mean_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
