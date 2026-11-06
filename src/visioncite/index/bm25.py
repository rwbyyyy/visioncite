"""BM25 检索：稀疏检索的工业默认值。

公式（Robertson & Zaragoza, 2009）：

    score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

逐项解释（面试常问）：
- ``tf`` 饱和：一个词出现 10 次并不比 3 次重要 3 倍，k1 控制饱和速度；
- ``b``：文档越长越容易"碰巧命中"，用长度归一压制；b=0 不归一，b=1 完全归一；
- ``IDF`` 用 ln(1+…) 形式，保证非负（经典 BM25 在极端 df 时可能为负）；
- 我们查询很短（一句话），所以不做 query tf 饱和。

实现上刻意保持"朴素倒排"：词 → [(doc, tf)]，检索时对查询词取并集。
没有谁比谁聪明，但每一步都能单独测、单独讲。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from visioncite.index.tokenize import tokenize


@dataclass
class BM25Index:
    """内存倒排索引。

    Attributes:
        k1: 词频饱和参数。
        b: 长度归一参数。
        postings: 词 → {文档下标: 词频}。
        doc_lengths: 每篇文档的 token 数。
        chunk_ids: 下标 → chunk_id，保证检索结果可回查。
    """

    k1: float = 1.5
    b: float = 0.75
    postings: dict[str, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    doc_lengths: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """建索引。

        Args:
            documents: ``[(chunk_id, text), ...]``；空文本会被跳过（但保留占位）。
        """
        self.postings = defaultdict(dict)
        self.doc_lengths = []
        self.chunk_ids = []

        for index, (chunk_id, text) in enumerate(documents):
            tokens = tokenize(text)
            self.chunk_ids.append(chunk_id)
            self.doc_lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term][index] = count

    @property
    def size(self) -> int:
        """文档数。"""
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        """检索。

        Args:
            query: 查询文本。
            top_k: 返回条数。

        Returns:
            ``[(chunk_id, score), ...]``，按分数降序；无命中返回空列表。
        """
        terms = tokenize(query)
        if not terms or self.size == 0:
            return []

        avgdl = sum(self.doc_lengths) / self.size or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):  # 去重：同一查询词只累加一次 IDF
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self._idf(len(posting))
            for doc_index, tf in posting.items():
                scores[doc_index] += idf * self._tf_component(tf, self.doc_lengths[doc_index], avgdl)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.chunk_ids[doc_index], score) for doc_index, score in ranked]

    def _idf(self, df: int) -> float:
        """计算 IDF（带 +1 平滑，保证非负）。

        Args:
            df: 包含该词的文档数。

        Returns:
            IDF 值。
        """
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def _tf_component(self, tf: int, doc_length: int, avgdl: float) -> float:
        """BM25 的词频项。

        Args:
            tf: 词频。
            doc_length: 文档长度。
            avgdl: 平均文档长度。

        Returns:
            饱和且长度归一后的词频得分。
        """
        length_norm = 1.0 - self.b + self.b * (doc_length / avgdl)
        return tf * (self.k1 + 1.0) / (tf + self.k1 * length_norm)
