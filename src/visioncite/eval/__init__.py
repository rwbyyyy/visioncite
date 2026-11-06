"""评测模块：合成语料、指标、bootstrap、命令行入口。"""

from visioncite.eval.corpus_gen import generate_corpus, load_or_generate
from visioncite.eval.dataset import QAItem, load_items, save_items
from visioncite.eval.metrics import (
    EvalRecord,
    MetricsSummary,
    answer_contains,
    citation_hit,
    citation_precision,
    exact_match,
    expected_calibration_error,
    summarize,
    token_f1,
)

__all__ = [
    "EvalRecord",
    "MetricsSummary",
    "QAItem",
    "answer_contains",
    "citation_hit",
    "citation_precision",
    "exact_match",
    "expected_calibration_error",
    "generate_corpus",
    "load_items",
    "load_or_generate",
    "save_items",
    "summarize",
    "token_f1",
]
