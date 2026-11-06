"""排序融合与多样性重排。

RRF（Reciprocal Rank Fusion）：
    不同检索器的分数不可比（BM25 分数无上界，余弦相似度在 [0,1]），
    直接加权求和需要调权重且对离群分数敏感。
    RRF 只用**名次**：score = Σ 1/(k + rank)，k 常取 60。
    它没有可训练参数，鲁棒性来自"名次比分数稳定"这一经验观察。

MMR（Maximal Marginal Relevance）：
    只按相关性排序会返回一堆几乎相同的证据（同一段的上下两块）。
    MMR 在每轮选择时同时考虑"与查询的相关性"和"与已选证据的差异"：
        MMR = λ · sim(q, d) - (1-λ) · max sim(d, selected)
    λ=1 退化为纯相关性，λ=0.7 是常用的折中。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """融合多路检索结果。

    Args:
        rankings: 每路检索的 chunk_id 有序列表（最相关在前）。
        k: 平滑常数，越大越削弱头部名次优势。
        weights: 每路权重；None 表示等权。

    Returns:
        ``[(chunk_id, fused_score), ...]``，按分数降序。
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] += weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """两个向量的余弦相似度（已归一化时等价于点积）。

    Args:
        a: 向量 A。
        b: 向量 B。

    Returns:
        相似度；任一向量模长为 0 时返回 0。
    """
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def mmr_select(
    chunk_ids: list[str],
    relevance: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    k: int,
    lambda_: float = 0.7,
) -> list[str]:
    """用 MMR 选出既相关又多样的证据。

    Args:
        chunk_ids: 候选 id。
        relevance: 每个候选与查询的相关性分数（一维）。
        embeddings: 候选向量矩阵，与 ``chunk_ids`` 对齐。
        k: 选出的数量。
        lambda_: 相关性权重，越大越偏相关性。

    Returns:
        选出的 id 列表，按选择顺序。
    """
    if k <= 0 or not chunk_ids:
        return []

    remaining = list(range(len(chunk_ids)))
    selected: list[int] = []
    relevance_list = [float(value) for value in relevance]

    while remaining and len(selected) < k:
        best_index = max(
            remaining,
            key=lambda i: _mmr_score(i, selected, relevance_list, embeddings, lambda_),
        )
        selected.append(best_index)
        remaining.remove(best_index)
    return [chunk_ids[index] for index in selected]


def _mmr_score(
    index: int,
    selected: list[int],
    relevance: list[float],
    embeddings: NDArray[np.float32],
    lambda_: float,
) -> float:
    """计算候选的 MMR 分数。

    Args:
        index: 候选下标。
        selected: 已选下标。
        relevance: 相关性列表。
        embeddings: 向量矩阵。
        lambda_: 相关性权重。

    Returns:
        MMR 分数。
    """
    if not selected:
        return relevance[index]
    redundancy = max(cosine_similarity(embeddings[index], embeddings[j]) for j in selected)
    return lambda_ * relevance[index] - (1.0 - lambda_) * redundancy
