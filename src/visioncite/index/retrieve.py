"""检索：文本与视觉两路召回，再用 RRF 融合。

为什么两路？
    BM25 擅长"关键词精确命中"（金额、合同编号），
    CLIP 擅长"语义/视觉相近"（一张柱状图、一个流程图）。
    两路各有盲区，融合后的召回率高于任何单路 —— 消融实验里会给出数字。

返回的分数统一归一化到 0~1，方便上层做阈值判断；
但排序本身仍然基于 RRF 名次，避免"分数尺度不同"带来的偏差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visioncite.config import Settings
from visioncite.index.fuse import mmr_select, reciprocal_rank_fusion
from visioncite.logging import get_logger
from visioncite.pipeline import Document
from visioncite.types import Chunk

logger = get_logger("index.retrieve")


@dataclass(frozen=True)
class Retrieved:
    """一条检索结果。

    Attributes:
        chunk: 命中的块。
        score: 归一化到 0~1 的融合分数。
    """

    chunk: Chunk
    score: float


def retrieve(document: Document, question: str, settings: Settings) -> list[Retrieved]:
    """检索与问题最相关的证据块。

    Args:
        document: 已建索引的文档。
        question: 用户问题。
        settings: 配置（top_k、RRF k、MMR λ）。

    Returns:
        按分数降序的证据列表；无命中时为空。
    """
    text_hits = document.bm25.search(question, top_k=settings.top_k * 3)
    visual_hits = _visual_search(document, question, settings)
    if not text_hits and not visual_hits:
        return []

    rankings: list[list[str]] = [[chunk_id for chunk_id, _ in text_hits]]
    if visual_hits:
        rankings.append([chunk_id for chunk_id, _ in visual_hits])
    fused = reciprocal_rank_fusion(rankings, k=settings.rrf_k)
    selected = _diversify(document, fused, settings)

    top = fused[0][1] if fused else 1.0
    results: list[Retrieved] = []
    for chunk_id in selected:
        chunk = document.chunk_by_id(chunk_id)
        if chunk is None:
            continue
        raw = dict(fused).get(chunk_id, 0.0)
        results.append(Retrieved(chunk=chunk, score=min(1.0, raw / top) if top else 0.0))
    return results


def _visual_search(
    document: Document,
    question: str,
    settings: Settings,
) -> list[tuple[str, float]]:
    """视觉召回：问题文本与块向量做余弦相似度。

    Args:
        document: 文档。
        question: 问题。
        settings: 配置。

    Returns:
        ``[(chunk_id, similarity), ...]``；视觉不可用时为空。
    """
    if document.vectors is None or document.encoder is None:
        return []
    encoder = document.encoder
    try:
        query_vector = encoder.encode_texts([question])[0]
    except Exception:  # noqa: BLE001 - 视觉失败不影响文本检索
        logger.exception("查询向量编码失败")
        return []
    similarities = document.vectors @ query_vector
    order = np.argsort(-similarities)[: settings.top_k * 3]
    return [(document.chunks[int(i)].chunk_id, float(similarities[int(i)])) for i in order]


def _diversify(
    document: Document,
    fused: list[tuple[str, float]],
    settings: Settings,
) -> list[str]:
    """用 MMR 去冗余；没有向量时退化为截断。

    Args:
        document: 文档。
        fused: RRF 结果。
        settings: 配置。

    Returns:
        选中的 chunk_id 列表。
    """
    candidates = [chunk_id for chunk_id, _ in fused]
    if document.vectors is None or len(candidates) <= 1:
        return candidates[: settings.top_k]

    index_by_id = {chunk.chunk_id: i for i, chunk in enumerate(document.chunks)}
    rows = [index_by_id[chunk_id] for chunk_id in candidates if chunk_id in index_by_id]
    if not rows:
        return candidates[: settings.top_k]
    relevance = np.array([dict(fused)[document.chunks[i].chunk_id] for i in rows], dtype=np.float32)
    embeddings = document.vectors[rows]
    try:
        chosen = mmr_select(
            [document.chunks[i].chunk_id for i in rows],
            relevance,
            embeddings,
            k=settings.top_k,
            lambda_=settings.mmr_lambda,
        )
    except Exception:  # noqa: BLE001 - 多样性重排失败就用原顺序
        logger.exception("MMR 失败，退回 RRF 顺序")
        return candidates[: settings.top_k]
    return chosen
