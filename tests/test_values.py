"""数值收敛测试：整行 → 用户要的那个值。"""

from __future__ import annotations

from visioncite.answer.values import asks_for_value, extract_value


def test_detects_value_questions() -> None:
    """问"多少/成绩/金额"才算数值问题。"""
    assert asks_for_value("计算机网络百分制成绩多少")
    assert asks_for_value("打印机的单价是多少")
    assert not asks_for_value("开票日期是哪一天")


def test_scores_prefer_integer_in_0_100() -> None:
    """成绩题：跳过 ID 与绩点，选 0~100 的整数。"""
    assert extract_value("计算机网络百分制成绩多少", "计算机网络 | -0014455-1 | 88 | 4.5 | 4.2") == "88"


def test_header_drives_column_choice() -> None:
    """有表头时按列名取，而不是"第一个数字"。"""
    header = "产品 | 数量 | 单价 | 小计"
    assert extract_value("打印机的单价是多少", "打印机 | 2 | 643.20 | 1286.40", header) == "643.20"
    assert extract_value("显示器卖了多少台", "显示器 | 12 | 899.00 | 10788.00", header) == "12"


def test_id_like_numbers_are_skipped() -> None:
    """编号/学号不能被当成答案。"""
    assert extract_value("这件商品数量多少", "商品 | -0014455-1 | 7") == "7"


def test_inline_amount() -> None:
    """没有表格竖线时，从整句里取数字（保留单位，便于阅读）。"""
    assert extract_value("合计金额是多少元", "合计金额：1,286.40 元。") == "1,286.40 元"


def test_returns_none_when_no_number() -> None:
    """没有数字就返回 None，交给调用方保留原答案。"""
    assert extract_value("备注写了什么", "备注：验收后付款") is None
