"""评测工具测试：指标函数、bootstrap、数据集、语料生成。"""

from __future__ import annotations

from pathlib import Path

import pytest

from visioncite.eval.bootstrap import bootstrap_mean_ci, paired_bootstrap_ci
from visioncite.eval.corpus_gen import find_cjk_font
from visioncite.eval.dataset import QAItem, load_items, save_items
from visioncite.eval.metrics import (
    EvalRecord,
    answer_contains,
    citation_precision,
    exact_match,
    expected_calibration_error,
    percentile,
    summarize,
    token_f1,
)
from visioncite.types import BBox, Citation


def _has_font() -> bool:
    """检查本机是否有可用中文字体（决定是否跑语料生成测试）。"""
    try:
        find_cjk_font()
    except Exception:  # noqa: BLE001
        return False
    return True


def test_exact_match_ignores_punctuation_and_space() -> None:
    """标点与空白不影响完全匹配。"""
    assert exact_match("合计金额：1,286.40 元", "合计金额1286.40元") == 1.0


def test_answer_contains() -> None:
    """包含关系比 EM 宽松。"""
    assert answer_contains("合计金额：1,286.40 元", "1,286.40") == 1.0
    assert answer_contains("开票日期：2026-10-25", "1286.40") == 0.0


def test_token_f1_partial() -> None:
    """部分重合时 F1 介于 0 和 1 之间。"""
    score = token_f1("合计金额：1286.40 元", "合计金额1286.40元")
    assert 0 < score <= 1


def test_token_f1_no_overlap() -> None:
    """完全不重合时为 0。"""
    assert token_f1("abc", "xyz") == 0.0


def test_citation_precision() -> None:
    """两个引用里一个命中，精确率为 0.5。"""
    gold = BBox(0, 0, 10, 10)
    good = Citation(chunk_id="a", region_id="p0-r0", page=0, bbox=BBox(0, 0, 10, 10), snippet="")
    bad = Citation(chunk_id="b", region_id="p0-r1", page=0, bbox=BBox(50, 50, 60, 60), snippet="")
    assert citation_precision([good, bad], gold) == 0.5


def test_ece_zero_when_perfectly_calibrated() -> None:
    """信心=1 且全对时 ECE 为 0。"""
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)


def test_ece_positive_when_overconfident() -> None:
    """信心 1 但全错时 ECE 接近 1。"""
    assert expected_calibration_error([1.0, 1.0], [False, False]) == pytest.approx(1.0)


def test_percentile_interpolates() -> None:
    """分位数用线性插值。"""
    assert percentile([0, 10], 50) == pytest.approx(5.0)


def test_summarize_counts_abstention() -> None:
    """弃答不算答错，但会影响答题率与弃答准确率。"""
    records = [
        EvalRecord("a", "invoice", True, False, "1", "1", 0.9, 1.0, contains=1.0),
        EvalRecord("b", "invoice", True, True, "", "2", 0.0, 1.0),
        EvalRecord("c", "chart", False, True, "", "", 0.0, 1.0),
    ]
    summary = summarize(records)
    assert summary.accuracy == pytest.approx(1.0)
    assert summary.coverage == pytest.approx(1 / 3)
    assert summary.abstain_accuracy == pytest.approx(2 / 3)


def test_bootstrap_ci_contains_mean() -> None:
    """bootstrap 区间应覆盖样本均值。"""
    low, high = bootstrap_mean_ci([1.0, 0.0, 1.0, 0.0, 1.0], n_boot=200, seed=1)
    assert low <= 0.6 <= high


def test_paired_bootstrap_detects_difference() -> None:
    """明显不同的两组，差值区间应不含 0。"""
    low, high = paired_bootstrap_ci([1.0] * 20, [0.0] * 20, n_boot=200, seed=2)
    assert low > 0 or high < 0


def test_paired_bootstrap_rejects_length_mismatch() -> None:
    """长度不一致要报错，避免静默对齐错误。"""
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 0.0])


def test_dataset_roundtrip(tmp_path: Path) -> None:
    """JSONL 写入后能原样读回。"""
    item = QAItem(
        item_id="x-1",
        doc_file="images/x.png",
        question="Q?",
        answer="A",
        citation=BBox(1, 2, 3, 4),
        answerable=True,
        category="invoice",
    )
    path = tmp_path / "items.jsonl"
    save_items(path, [item])
    loaded = load_items(path)
    assert loaded == [item]


@pytest.mark.skipif(not _has_font(), reason="缺少中文字体，跳过语料生成测试")
def test_generate_corpus(tmp_path: Path) -> None:
    """生成 1 份/类，检查图片与样本数量。"""
    from visioncite.eval.corpus_gen import generate_corpus

    corpus = generate_corpus(tmp_path / "corpus", per_template=1)
    assert len(corpus.images) == 4
    assert all(path.is_file() for path in corpus.images)
    assert len(corpus.items) >= 12
    assert sum(1 for item in corpus.items if not item.answerable) >= 1
