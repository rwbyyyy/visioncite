"""索引与检索测试：分词、BM25、RRF/MMR、切块。"""

from __future__ import annotations

import numpy as np
import pytest

from visioncite.index.bm25 import BM25Index
from visioncite.index.chunking import chunk_region, split_sentences
from visioncite.index.fuse import mmr_select, reciprocal_rank_fusion
from visioncite.index.tokenize import normalize, tokenize
from visioncite.types import BBox, Page, Region


def test_normalize_fullwidth_and_thousands() -> None:
    """全角转半角、千分位去掉、统一小写。"""
    assert normalize("１，２８６.４０") == "1286.40"
    assert normalize("ABC") == "abc"


def test_tokenize_chinese_bigrams() -> None:
    """中文按字 bigram 切分。"""
    assert tokenize("合计金额") == ["合计", "计金", "金额"]


def test_tokenize_splits_letters_and_numbers() -> None:
    """OCR 常见的粘连文本要被拆开，否则检索不到。"""
    assert tokenize("ECE0.08") == ["ece", "0.08"]


def test_tokenize_single_chinese_char() -> None:
    """单字保留（否则"月"这种查询永远无解）。"""
    assert tokenize("月") == ["月"]


def _index() -> BM25Index:
    """三篇小文档的索引。"""
    index = BM25Index()
    index.fit(
        [
            ("doc1", "合计金额：1286.40 元"),
            ("doc2", "开票日期：2026-10-25"),
            ("doc3", "采购发票 备注：验收后付款"),
        ]
    )
    return index


def test_bm25_ranks_matching_doc_first() -> None:
    """查询词命中哪篇，哪篇就排第一。"""
    results = _index().search("金额 多少")
    assert results[0][0] == "doc1"


def test_bm25_empty_query_returns_empty() -> None:
    """空查询不返回任何结果。"""
    assert _index().search("") == []


def test_bm25_idf_positive() -> None:
    """IDF 必须非负（ln(1+…) 形式的意义）。"""
    index = _index()
    assert index._idf(3) > 0


def test_bm25_scores_are_sorted() -> None:
    """返回结果按分数降序。"""
    scores = [score for _, score in _index().search("发票 金额")]
    assert scores == sorted(scores, reverse=True)


def test_rrf_boosts_items_in_both_rankings() -> None:
    """两路都出现的项应排第一，体现"共识"。"""
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert fused[0][0] == "b"


def test_rrf_weights_change_order() -> None:
    """给第二路更高权重时，第二路的头部应上升。"""
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[0.1, 1.0])
    assert fused[0][0] == "b"


def test_mmr_prefers_diverse_candidates() -> None:
    """两个几乎相同的向量只应选中一个。"""
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    relevance = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    chosen = mmr_select(["a", "b", "c"], relevance, embeddings, k=2, lambda_=0.5)
    assert "a" in chosen
    assert "c" in chosen


def test_split_sentences_keeps_punctuation() -> None:
    """切句后句末标点保留。"""
    assert split_sentences("第一句。第二句！第三句") == ["第一句。", "第二句！", "第三句"]


def test_chunk_region_produces_stable_ids() -> None:
    """块 id 由文档/区域/序号组成，方便回溯。"""
    page = Page(index=0, width=100, height=100, image=np.zeros((100, 100, 3), dtype=np.uint8))
    region = Region(region_id="p0-r1", page=0, bbox=BBox(0, 0, 100, 20), text="一句话。")
    chunks = chunk_region("doc", page, region)
    assert chunks[0].chunk_id == "doc:p0-r1:0"


def test_chunk_region_splits_long_text() -> None:
    """长文本会被切成多块，块数与长度相关。"""
    page = Page(index=0, width=100, height=100, image=np.zeros((100, 100, 3), dtype=np.uint8))
    text = "。".join(["这是一段用于测试切块逻辑的长文本"] * 10) + "。"
    region = Region(region_id="p0-r2", page=0, bbox=BBox(0, 0, 100, 20), text=text)
    assert len(chunk_region("doc", page, region, max_chars=60)) > 1


@pytest.mark.parametrize("text", ["", "   "])
def test_chunk_empty_region_returns_nothing(text: str) -> None:
    """空文本不产生块。"""
    page = Page(index=0, width=10, height=10, image=np.zeros((10, 10, 3), dtype=np.uint8))
    region = Region(region_id="p0-r0", page=0, bbox=BBox(0, 0, 10, 10), text=text)
    assert chunk_region("doc", page, region) == []
