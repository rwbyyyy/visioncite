"""答案校验：把"看起来对的答案"筛成"有证据支撑的答案"。

三条规则，全部可测试、可解释：

1. **引用必须存在**：模型可能编一个不存在的 chunk_id。
2. **数字必须能在证据里找到**：答案里的 ``1,286.40`` 必须出现在被引文本中。
   数字幻觉是最危险的一类错误（金额、日期、比例），一条规则挡住大部分。
3. **词面支撑度**：答案的 token 里至少 ``min_support`` 比例出现在证据文本中。
   它不能证明答案正确，但能挡住"答非所问"和模型自由发挥。

这套校验对 extractive 路径几乎总是通过（答案本来就是摘录的），
真正的价值在 VLM 路径：模型越自由，越需要这层网。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from visioncite.index.tokenize import normalize, tokenize
from visioncite.types import Chunk, Citation

# 数字：整数、小数、千分位、百分比、货币符号后的数
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*(?:\s*%)?")
_CURRENCY = re.compile(r"[¥$€£]")


@dataclass(frozen=True)
class Verification:
    """校验结果。

    Attributes:
        ok: 是否全部通过。
        reasons: 失败原因（中文，可直接写日志）。
        checks: 每项检查的明细，便于调试与展示。
    """

    ok: bool
    reasons: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)


def numbers_in(text: str) -> list[str]:
    """抽取文本中的数字并归一化（去掉千分位与货币符号）。

    Args:
        text: 文本。

    Returns:
        数字字符串列表，例如 ``["1286.40", "5%"]``。
    """
    cleaned = _CURRENCY.sub(" ", normalize(text))
    numbers: list[str] = []
    for match in _NUMBER.finditer(cleaned):
        value = match.group(0).replace(",", "").replace(" ", "")
        numbers.append(value)
    return numbers


def numbers_supported(answer: str, evidence_texts: list[str]) -> bool:
    """答案里的每个数字都能在证据中找到。

    Args:
        answer: 答案文本。
        evidence_texts: 被引证据的文本。

    Returns:
        True 表示全部数字都有出处（答案没有数字时也返回 True）。
    """
    evidence_numbers: set[str] = set()
    for text in evidence_texts:
        evidence_numbers.update(numbers_in(text))
    return all(number in evidence_numbers for number in numbers_in(answer))


def lexical_support(answer: str, evidence_texts: list[str]) -> float:
    """答案 token 被证据覆盖的比例。

    Args:
        answer: 答案文本。
        evidence_texts: 证据文本。

    Returns:
        0~1；答案为空时返回 0。
    """
    answer_tokens = [token for token in tokenize(answer) if not token.isdigit()]
    if not answer_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for text in evidence_texts:
        evidence_tokens.update(tokenize(text))
    hits = sum(1 for token in answer_tokens if token in evidence_tokens)
    return hits / len(answer_tokens)


def citation_ids_exist(citations: list[Citation], chunks: list[Chunk]) -> bool:
    """每条引用都指向真实存在的块。

    Args:
        citations: 引用列表。
        chunks: 可用的块。

    Returns:
        True 表示全部有效。
    """
    known = {chunk.chunk_id for chunk in chunks}
    return all(citation.chunk_id in known for citation in citations)


def verify_answer(
    answer: str,
    citations: list[Citation],
    chunks: list[Chunk],
    min_support: float = 0.5,
) -> Verification:
    """执行全部校验。

    Args:
        answer: 答案文本。
        citations: 引用。
        chunks: 可用块（用于查引用 id 与证据文本）。
        min_support: 词面支撑度下限。

    Returns:
        Verification。
    """
    reasons: list[str] = []
    if not answer.strip():
        reasons.append("答案为空")

    if not citation_ids_exist(citations, chunks):
        reasons.append("引用了不存在的证据块")

    evidence_texts = _evidence_texts(citations, chunks)
    if not numbers_supported(answer, evidence_texts):
        reasons.append("答案中的数字不在证据里")

    support = lexical_support(answer, evidence_texts)
    if support < min_support:
        reasons.append(f"词面支撑不足：{support:.2f} < {min_support:.2f}")

    checks = {
        "has_answer": bool(answer.strip()),
        "citations_exist": not any("引用" in reason for reason in reasons),
        "numbers_grounded": not any("数字" in reason for reason in reasons),
        "lexical_support": support >= min_support,
    }
    return Verification(ok=not reasons, reasons=tuple(reasons), checks=checks)


def _evidence_texts(citations: list[Citation], chunks: list[Chunk]) -> list[str]:
    """把引用还原成证据原文。

    Args:
        citations: 引用。
        chunks: 块列表。

    Returns:
        证据文本列表（引用失效的会被跳过）。
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [by_id[citation.chunk_id].text for citation in citations if citation.chunk_id in by_id]
