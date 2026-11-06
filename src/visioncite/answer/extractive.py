"""Extractive 答案器：不调用大模型也能给出"有出处的一句话"。

做法：
1. 把每条证据按句切开；
2. 用问题的 token 覆盖率给句子打分（命中越多、句子越短，分越高）；
3. 取最高分句子作为答案，并把它所在的块作为引用；
4. 用"覆盖率 + 领先幅度"估一个置信度。

为什么要有它？
    - 离线、无 API key、毫秒级，CI 和演示都能跑；
    - 它是 VLM 路径的**对照基线**：如果 VLM 只比它高一点，说明大部分收益来自检索；
    - 面试时能讲清楚"答案质量 = 检索质量 × 抽取质量"，而不是"模型很强"。

局限：只能摘录现成句子，不能跨段归纳、不能做算术。这些交给可选的 VLM 后端。
"""

from __future__ import annotations

from dataclasses import dataclass

from visioncite.index.tokenize import tokenize
from visioncite.types import Chunk

# 疑问词/虚词：切 bigram 之前先把这些**字符**从问题里剔除。
# 为什么按字符删而不是按词删？中文没有空格，"是多少" 会被切成
# "是多/多少" 两个 bigram，按词表根本删不干净；按字符删最直接。
_QUESTION_CHARS = set("的是了在和与或及多少什么怎么如何哪请问告诉一下可以吗呢吧啊呀几条件个这那些有")

# 去掉疑问字符后仍然没有信息量的 bigram
STOPWORDS = frozenset(
    {
        "我们", "你们", "他们", "这个", "那个", "这些", "那些", "其中",
        "以及", "还有", "然后", "并且", "或者", "因为", "所以", "如果",
        "虽然", "但是", "而且", "已经", "正在", "一直", "非常", "比较",
        "the", "is", "are", "was", "what", "how", "much", "many", "of",
        "in", "on", "and", "for", "with", "to", "a", "an", "it", "this", "that",
    }
)


@dataclass(frozen=True)
class Candidate:
    """一个候选答案句。

    Attributes:
        text: 句子原文。
        chunk: 所属证据块。
        score: 句子得分。
        coverage: 问题 token 被覆盖的比例。
    """

    text: str
    chunk: Chunk
    score: float
    coverage: float


def query_terms(question: str) -> list[str]:
    """提取问题里的有效 token。

    两步：
    1. 删除疑问字符（是/多/少/什/么/怎/么…），把"合计金额是多少元"
       变成"合计金额元"，避免疑问句式主导覆盖率；
    2. 切 bigram 后去重、去停用词。

    Args:
        question: 用户问题。

    Returns:
        去重后的 token 列表。
    """
    stripped = "".join(char for char in question if char not in _QUESTION_CHARS)
    terms = [term for term in tokenize(stripped) if term not in STOPWORDS]
    # 单字（非英文/数字）区分度太低，丢掉
    filtered = [term for term in terms if len(term) > 1 or term.isascii()]
    return list(dict.fromkeys(filtered))


def score_sentence(sentence: str, terms: list[str]) -> tuple[float, float]:
    """给句子打分。

    打分公式：
        coverage = 命中的问题词数 / 问题词总数
        density  = 命中次数 / sqrt(句子长度)
        score    = 0.7 * coverage + 0.3 * density
    除以 sqrt 长度是为了避免"长句子碰巧命中更多词"。

    Args:
        sentence: 句子。
        terms: 问题词。

    Returns:
        ``(score, coverage)``。
    """
    if not terms:
        return 0.0, 0.0
    sentence_tokens = tokenize(sentence)
    token_set = set(sentence_tokens)
    hits = sum(1 for term in terms if term in token_set)
    coverage = hits / len(terms)
    density = hits / max(1.0, len(sentence_tokens) ** 0.5)
    return 0.7 * coverage + 0.3 * min(1.0, density), coverage


def rank_candidates(question: str, evidence: list[Chunk]) -> list[Candidate]:
    """对所有证据句打分排序。

    Args:
        question: 问题。
        evidence: 证据块（通常来自检索）。

    Returns:
        候选句列表，按分数降序。
    """
    terms = query_terms(question)
    candidates: list[Candidate] = []
    for chunk in evidence:
        for sentence in _sentences(chunk.text):
            score, coverage = score_sentence(sentence, terms)
            if score <= 0:
                continue
            candidates.append(Candidate(text=sentence, chunk=chunk, score=score, coverage=coverage))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _sentences(text: str) -> list[str]:
    """切句（复用切块逻辑，保持行为一致）。

    Args:
        text: 文本。

    Returns:
        句子列表。
    """
    from visioncite.index.chunking import split_sentences

    return split_sentences(text) or [text]


def compose_answer(question: str, evidence: list[Chunk]) -> tuple[str, list[Chunk], float]:
    """生成答案、引用与置信度。

    Args:
        question: 问题。
        evidence: 证据块，按检索分排序。

    Returns:
        ``(answer, cited_chunks, confidence)``；无候选时返回空答案与 0 置信度。
    """
    candidates = rank_candidates(question, evidence)
    if not candidates:
        return "", [], 0.0

    best = candidates[0]
    margin = _margin(candidates)
    confidence = _confidence(best.coverage, margin, best.score)
    cited = _cited_chunks(best, candidates)
    return best.text.strip(), cited, confidence


def _margin(candidates: list[Candidate]) -> float:
    """计算第一与第二名的相对差距（0~1）。

    Args:
        candidates: 已排序候选。

    Returns:
        相对差距；只有一个候选时为 1。
    """
    if len(candidates) == 1 or candidates[0].score <= 0:
        return 1.0
    return max(0.0, (candidates[0].score - candidates[1].score) / candidates[0].score)


def _confidence(coverage: float, margin: float, score: float) -> float:
    """把三个信号压成 0~1 的置信度。

    权重不是拍脑袋：覆盖率决定"是否答在点上"，margin 决定"是否有歧义"，
    score 只作为弱信号。后续评测会用 ECE 检查这个估计是否过自信。

    Args:
        coverage: 问题词覆盖率。
        margin: 领先幅度。
        score: 句子得分。

    Returns:
        0~1 的置信度。
    """
    value = 0.55 * coverage + 0.25 * margin + 0.20 * min(1.0, score)
    return float(max(0.0, min(1.0, value)))


def _cited_chunks(best: Candidate, candidates: list[Candidate]) -> list[Chunk]:
    """选出要展示的引用块：最佳块 + 另一个不同块的次优句。

    Args:
        best: 最佳候选。
        candidates: 全部候选。

    Returns:
        最多两个块，按出现顺序。
    """
    cited = [best.chunk]
    for candidate in candidates[1:]:
        if candidate.chunk.chunk_id != best.chunk.chunk_id:
            cited.append(candidate.chunk)
            break
    return cited
