"""切块：把区域文本切成适合检索的小段。

为什么需要切块？
    BM25 对长文档的分数会被"长度归一"压扁，一段 2000 字的区域里
    只有一句话相关时，检索分数会明显低于短而精准的块。
    经验上中文 150~300 字一块，既能保留上下文，又不会稀释信号。

做法很朴素（也足够好解释）：
    按句末标点切句 → 顺序累加，超过 ``max_chars`` 就落一块 → 不重叠。
    长表格行不会被切开，因为表格按行拼接后用换行分隔，按行切更自然。
"""

from __future__ import annotations

import re

from visioncite.types import Chunk, Page, Region

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|(?<=\n)")


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句。

    Args:
        text: 区域文本。

    Returns:
        句子列表（保留标点，去掉纯空白项）。
    """
    parts = [part.strip() for part in _SENTENCE_END.split(text)]
    return [part for part in parts if part]


def chunk_region(doc_id: str, page: Page, region: Region, max_chars: int = 240) -> list[Chunk]:
    """把一个区域切成若干检索块。

    Args:
        doc_id: 文档 id。
        page: 所属页。
        region: 区域。
        max_chars: 单块最大字符数。

    Returns:
        Chunk 列表；区域无文本时返回空列表。
    """
    if not region.text.strip():
        return []

    # 表格按"行"切，且**不要**把多行拼进同一个块：
    # 表格行没有句末标点，_pack 会把相邻行直接粘在一起（"小计显示器"），
    # 检索和答案都会变成一大片单元格文本。
    if region.kind == "table":
        pieces = _rows_as_chunks(region.text.splitlines(), max_chars)
    else:
        pieces = _pack(split_sentences(region.text), max_chars)
    pieces = pieces or [region.text.strip()]
    return [
        Chunk(
            chunk_id=f"{doc_id}:{region.region_id}:{index}",
            doc_id=doc_id,
            page=page.index,
            region_id=region.region_id,
            bbox=region.bbox,
            kind=region.kind,
            text=piece,
        )
        for index, piece in enumerate(pieces)
    ]


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    """把句子贪心装箱。

    两种情况要区分：
    - 正常句子：按顺序累加，超限换块；
    - 超长句子（例如 OCR 把整页连成一段、没有句号）：直接按字符数硬切，
      否则会产生一个"包住整页"的块，检索分数被稀释、答案也会过长。

    Args:
        sentences: 句子列表。
        max_chars: 单块上限。

    Returns:
        文本块列表。
    """
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(_hard_split(sentence, max_chars))
            continue
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _rows_as_chunks(rows: list[str], max_chars: int) -> list[str]:
    """表格行 → 检索块：一行一块，超长行再硬切。

    Args:
        rows: 表格文本行。
        max_chars: 单块最大字符数。

    Returns:
        文本块列表。
    """
    chunks: list[str] = []
    for row in rows:
        text = row.strip()
        if not text:
            continue
        chunks.extend(_hard_split(text, max_chars) if len(text) > max_chars else [text])
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """按固定长度硬切文本（只在没有标点可切时使用）。

    Args:
        text: 文本。
        max_chars: 每段最大长度。

    Returns:
        文本段列表。
    """
    windows = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    return [window for window in windows if window.strip()]


def chunks_from_pages(doc_id: str, pages: list[Page], max_chars: int = 240) -> list[Chunk]:
    """遍历所有页面生成检索块。

    Args:
        doc_id: 文档 id。
        pages: 页面列表。
        max_chars: 单块上限。

    Returns:
        全部块。
    """
    chunks: list[Chunk] = []
    for page in pages:
        for region in page.regions:
            chunks.extend(chunk_region(doc_id, page, region, max_chars))
    return chunks
