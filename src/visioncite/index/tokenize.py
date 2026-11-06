"""分词：中文按字 bigram，英文/数字按词。

为什么不用 jieba？
    1. 引入一个词典依赖，冷启动和内存都要成本；
    2. 我们要检索的是**短问题 vs 短文本块**，字 bigram 对未登录词、专有名词更稳；
    3. 更好解释：``金额`` → ``金额``，``总金额`` → ``总金`` + ``金额``，
       即使分词器把"总金额"切成别的词，bigram 仍有命中。

代价：
    - 索引规模约为字数的 1 倍（可接受）；
    - 对"的/了"这类高频虚词没有停用词表，靠 BM25 的 IDF 自动降权。

英文与数字统一小写；千分位逗号先去掉，避免 "1,286" 与 "1286" 检索不上。
"""

from __future__ import annotations

import re
import unicodedata

# 中日韩统一表意文字基本区；够覆盖常见中文文档
_CJK = "\u4e00-\u9fff"
# 英文按字母、数字按数值分开切：
# OCR 经常把 "ECE0.08" 连在一起，如果按字母数字混合切，
# "ECE" 和 "0.08" 都会变成检索不到的长 token。
_TOKEN_PATTERN = re.compile(rf"[{_CJK}]+|[a-z]+|[0-9]+(?:\.[0-9]+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize(text: str) -> str:
    """文本归一化：全角转半角、去千分位、转小写、压缩空白。

    NFKC 会把全角数字/字母/标点转成半角，解决"１，２８６"这类 OCR 结果。

    Args:
        text: 原始文本。

    Returns:
        归一化后的文本。
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _THOUSANDS.sub("", normalized)
    normalized = normalized.lower()
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    """把文本切成检索词。

    规则：
    - 连续中文 → 长度为 2 的滑窗（bigram），单字单独成词；
    - 连续英数 → 整体作为一词，保留 ``. - _ / %`` 连接的复合词（如 3.5%、v2.1）。

    Args:
        text: 原始文本。

    Returns:
        token 列表（可能重复，重复次数即词频）。
    """
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalize(text)):
        piece = match.group(0)
        if _is_cjk(piece):
            tokens.extend(_cjk_tokens(piece))
        else:
            tokens.append(piece)
    return tokens


def _is_cjk(piece: str) -> bool:
    """判断片段是否为中文。

    Args:
        piece: 片段。

    Returns:
        True 表示以中文为主。
    """
    return bool(piece) and "\u4e00" <= piece[0] <= "\u9fff"


def _cjk_tokens(piece: str) -> list[str]:
    """中文片段转 bigram（长度为 1 时保留单字）。

    Args:
        piece: 连续中文片段。

    Returns:
        token 列表。
    """
    if len(piece) == 1:
        return [piece]
    return [piece[i : i + 2] for i in range(len(piece) - 1)]
