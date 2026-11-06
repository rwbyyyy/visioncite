"""数值抽取：把"一整行"收敛成"用户真正要的那个数"。

问题背景：
    抽取式答案返回的是证据句。表格行往往长这样：
        ``计算机网络 | -0014455-1 | 88 | 4.5 | 4.2``
    用户问"百分制成绩多少"，他要的是 ``88``，不是整行。

做法（三步，全部可解释）：
    1. 用 ``|`` 切单元格；没有 ``|`` 就按空白/标点切"数字候选"；
    2. 如果拿到了表头（同区域的第 0 块），用**列名对齐问题里的词**
       （"单价"→单价列、"百分制"→百分制列），直接取那一列；
    3. 没有表头就按语义启发式挑：先剔除 ID 类数字（``-0014455-1``），
       再按"百分制/成绩"偏好 0~100 的整数、按"金额/合计"偏好小数。

为什么不用大模型：
    这是一条**可测试的规则**：每一列怎么选、为什么剔除 ID 都能写成用例。
    规则错的时候你能定位到具体分支，而不是"模型今天心情不好"。
"""

from __future__ import annotations

import re

# 整格就是一个数值（可带单位），例如 "88"、"1,286.40"、"5%"、"12 台"
_VALUE_CELL = re.compile(r"^-?\d+(?:[.,]\d+)*\s*(?:%|元|分|台|个|次|门|小时|人|万次|亿次)?$")
# 行内数字候选
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*\s*(?:%|元|分|台|个|次|门|小时|人|万次|亿次)?")
# OCR 经常把竖线识别成"丨/｜"，切单元格时三种都接受
_CELL_SPLIT = re.compile(r"[|丨｜]")


def split_cells(text: str) -> list[str]:
    """把表格行切成单元格（兼容 OCR 的竖线变体）。

    Args:
        text: 表格行文本。

    Returns:
        去掉首尾空白的单元格列表。
    """
    return [cell.strip() for cell in _CELL_SPLIT.split(text)]

# 问题里出现这些词，说明用户想要一个具体值
VALUE_KEYWORDS = (
    "多少",
    "几",
    "成绩",
    "分数",
    "得分",
    "金额",
    "合计",
    "总额",
    "单价",
    "数量",
    "百分比",
    "百分制",
    "绩点",
    "学分",
)

# 列名同义词：表头写"单价"，用户可能问"多少钱"
_COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "单价": ("单价", "价格", "多少钱", "价位"),
    "数量": ("数量", "多少台", "多少个", "几台", "几个"),
    "百分制": ("百分制", "成绩", "分数", "得分"),
    "成绩": ("成绩", "分数", "得分"),
    "合计": ("合计", "总额", "总计"),
    "金额": ("金额", "多少钱", "合计"),
    "绩点": ("绩点", "gpa"),
    "学分": ("学分",),
}


def asks_for_value(question: str) -> bool:
    """问题是否在要一个具体数值。

    Args:
        question: 用户问题。

    Returns:
        True 表示可以做数值收敛。
    """
    lowered = question.lower()
    return any(keyword in lowered for keyword in VALUE_KEYWORDS)


def extract_value(question: str, sentence: str, header: str | None = None) -> str | None:
    """从证据句里抽出用户要的值。

    Args:
        question: 用户问题。
        sentence: 证据句（可能是 ``a | b | c`` 形式的表格行）。
        header: 同区域的表头行；没有就传 None。

    Returns:
        抽到的值（字符串，保留原格式）；抽不到返回 None。
    """
    cells = split_cells(sentence)
    if len(cells) >= 2:
        column = _match_column(question, header) if header else None
        if column is not None and column < len(cells) and _is_value_cell(cells[column]):
            return cells[column]
        return _pick_numeric(question, cells)
    return _pick_inline(question, sentence)


def _pick_inline(question: str, sentence: str) -> str | None:
    """没有表格分隔时，按"问题词的位置"挑数字。

    优先级（从强到弱）：
    1. 问题里有英文/数字锚点（如 ``ECE``、``GPA``）→ 取锚点后的第一个数字；
    2. 问题里的中文内容词（如"访问量""金额"）→ 取该词之后最近的数字；
    3. 带单位的数字（"21 万次""1,286.40 元"）优先于裸数字；
    4. 兜底取第一个数字。

    这样 ``10 月访问量：21 万次`` 不会被误取成 ``10``。

    Args:
        question: 用户问题。
        sentence: 证据句。

    Returns:
        选中的值；没有数字返回 None。
    """
    matches = [(match.start(), match.group(0)) for match in _NUMBER.finditer(sentence)]
    matches = [
        (start, value)
        for start, value in matches
        if not _embedded_in_code(sentence, start, value)
    ]
    if not matches:
        return None

    lowered = sentence.lower()
    for term in _anchors(question):
        position = lowered.find(term)
        if position < 0:
            continue
        after = [value for start, value in matches if start >= position]
        if after:
            return after[0]

    with_unit = [value for _, value in matches if _has_unit(value)]
    if with_unit:
        return with_unit[0]
    return matches[0][1]


def _anchors(question: str) -> list[str]:
    """提取问题里的"锚点词"：英文词优先，其次中文 bigram。

    Args:
        question: 用户问题。

    Returns:
        锚点列表（去重、保持出现顺序）。
    """
    english = re.findall(r"[A-Za-z]{2,}", question)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    bigrams: list[str] = []
    for segment in chinese:
        bigrams.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return list(dict.fromkeys([term.lower() for term in english] + bigrams))


def _has_unit(value: str) -> bool:
    """数字串是否带单位（元/万次/%/台…）。

    Args:
        value: 数字串。

    Returns:
        True 表示带单位。
    """
    return bool(re.search(r"[%元分台个次门人]|万次|亿次|小时", value))


def _match_column(question: str, header: str) -> int | None:
    """用表头把问题里的词对齐到列号。

    Args:
        question: 用户问题。
        header: 表头行。

    Returns:
        列下标；对不上返回 None。
    """
    columns = split_cells(header)
    for index, name in enumerate(columns):
        for alias in _COLUMN_SYNONYMS.get(name, (name,)):
            if alias and alias in question:
                return index
    return None


def _embedded_in_code(sentence: str, start: int, value: str) -> bool:
    """判断命中是否只是编号的一部分（如 ``INV-2026-100`` 里的 ``-2026``）。

    这类"数字"属于编码，不是用户要的答案；直接跳过，
    让上层保留整句答案（整句里含有完整编号）。

    Args:
        sentence: 证据句。
        start: 命中起点。
        value: 命中的数字串。

    Returns:
        True 表示这是编码的一部分。
    """
    end = start + len(value)
    before = sentence[start - 1] if start > 0 else ""
    after = sentence[end] if end < len(sentence) else ""
    # 前面是字母说明数字属于编码（INV-2026-100）；
    # 是数字且自身带负号，说明是编号里的分隔（-2026-）。
    if before.isalpha():
        return True
    if before.isalnum() and value.startswith("-"):
        return True
    # 后面紧跟字母/数字（中间没有空格）同样视为编码的一部分
    return bool(after.isalnum() and not value.endswith(" "))


def _pick_numeric(question: str, cells: list[str]) -> str | None:
    """按语义启发式挑一个数值。

    Args:
        question: 用户问题。
        cells: 候选单元格/数字串。

    Returns:
        选中的值；没有合适候选返回 None。
    """
    candidates = [cell for cell in cells if _is_value_cell(cell) and not _looks_like_id(cell)]
    if not candidates:
        return None

    wants_score = any(word in question for word in ("百分制", "成绩", "分数", "得分"))
    if wants_score:
        for cell in candidates:
            if _as_int(cell) is not None and 0 <= _as_int(cell) <= 100:  # type: ignore[operator]
                return cell
    return candidates[0]


def _is_value_cell(text: str) -> bool:
    """判断单元格是不是纯数值。

    Args:
        text: 单元格文本。

    Returns:
        True 表示是数值。
    """
    return bool(_VALUE_CELL.match(text.strip()))


def _looks_like_id(text: str) -> bool:
    """判断一个数值串是否像编号（学号/课程号），而不是成绩。

    规则：含连字符、或以 0 开头的长数字串，一律当 ID 排除。
    例：``-0014455-1``、``330015242210816``（15 位学号）。

    Args:
        text: 候选串。

    Returns:
        True 表示像编号。
    """
    stripped = text.replace(",", "").strip()
    if "-" in stripped:
        return True
    digits = re.sub(r"\D", "", stripped)
    return stripped.startswith("0") and len(digits) >= 6


def _as_int(text: str) -> int | None:
    """把纯整数格转成 int（小数/带单位返回 None）。

    Args:
        text: 单元格。

    Returns:
        int 或 None。
    """
    stripped = text.strip()
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    return None
