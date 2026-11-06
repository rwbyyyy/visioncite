"""评测指标：每一个都写清楚定义，避免"数字好看但口径不明"。

指标清单（报告与 README 使用同一套定义）：
- AnswerAcc：标准答案（归一化后）出现在预测答案中，视为答对；
- EM：归一化后完全相等（更严格，主要用于观察抽取粒度）；
- F1：token 级 P/R 的调和平均（中文按字 bigram，复用检索分词）；
- Cite-P / Cite-R：引用框与标准框 IoU>=0.5 视为命中；
- AbstainAcc：可答题不弃答、不可答题弃答的比例；
- ECE：置信度校准误差（10 个等宽分箱）；
- p50 / p95：延迟分位数。

重要口径：
    AnswerAcc / EM / F1 / Cite-P / Cite-R 只在**可答题且实际作答**的样本上统计
    （弃答不是答错，它是"没答"），答题率（coverage）与弃答准确率单独报告。
    这样读者能看出"少答换准确"的取舍，而不是被一个混合指标误导。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from visioncite.index.tokenize import tokenize
from visioncite.types import BBox, Citation


def normalize_text(text: str) -> str:
    """评测用归一化：去空白、去常见标点、统一小写。

    Args:
        text: 文本。

    Returns:
        归一化结果。
    """
    from visioncite.index.tokenize import normalize

    cleaned = normalize(text)
    for mark in "，。、；：！？,.;:!?“”\"'（）()【】[]《》<>":
        cleaned = cleaned.replace(mark, "")
    return cleaned.replace(" ", "")


def exact_match(prediction: str, gold: str) -> float:
    """完全匹配。

    Args:
        prediction: 预测答案。
        gold: 标准答案。

    Returns:
        1.0 或 0.0。
    """
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def answer_contains(prediction: str, gold: str) -> float:
    """标准答案是否出现在预测中。

    抽取式系统的答案往往是"包含答案的整句"，
    用 contains 比 EM 更贴近实际可用性；EM 同时保留，用来看抽取粒度。

    Args:
        prediction: 预测答案。
        gold: 标准答案。

    Returns:
        1.0 或 0.0。
    """
    gold_text = normalize_text(gold)
    if not gold_text:
        return 0.0
    return 1.0 if gold_text in normalize_text(prediction) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """token 级 F1。

    Args:
        prediction: 预测答案。
        gold: 标准答案。

    Returns:
        0~1。
    """
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    gold_left = list(gold_tokens)
    for token in pred_tokens:
        if token in gold_left:
            common += 1
            gold_left.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_hit(citations: list[Citation], gold: BBox | None, threshold: float = 0.5) -> bool:
    """是否有引用命中标准框。

    Args:
        citations: 预测引用。
        gold: 标准证据框。
        threshold: IoU 阈值。

    Returns:
        True 表示命中。
    """
    if gold is None:
        return False
    return any(citation.bbox.iou(gold) >= threshold for citation in citations)


def citation_precision(citations: list[Citation], gold: BBox | None, threshold: float = 0.5) -> float:
    """引用精确率：正确引用 / 全部引用。

    Args:
        citations: 引用。
        gold: 标准框。
        threshold: IoU 阈值。

    Returns:
        0~1；没有引用时为 0。
    """
    if not citations or gold is None:
        return 0.0
    hits = sum(1 for citation in citations if citation.bbox.iou(gold) >= threshold)
    return hits / len(citations)


def expected_calibration_error(
    confidences: list[float],
    corrects: list[bool],
    bins: int = 10,
) -> float:
    """期望校准误差 ECE。

    把置信度分成 B 个等宽区间，每箱计算 |平均置信度 - 实际准确率|，
    再按样本数加权：
        ECE = Σ_b (n_b / N) · |acc_b - conf_b|

    Args:
        confidences: 每个样本的信心值。
        corrects: 每个样本是否答对。
        bins: 分箱数。

    Returns:
        ECE（0 表示完美校准）。
    """
    if not confidences:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidences)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        indices = [
            i
            for i, (conf, _) in enumerate(zip(confidences, corrects, strict=True))
            if (low < conf <= high) or (low == 0 and conf == 0)
        ]
        if not indices:
            continue
        conf_mean = float(np.mean([confidences[i] for i in indices]))
        acc = float(np.mean([1.0 if corrects[i] else 0.0 for i in indices]))
        error += (len(indices) / total) * abs(acc - conf_mean)
    return error


def percentile(values: list[float], q: float) -> float:
    """分位数（线性插值）。

    Args:
        values: 数值列表。
        q: 0~100。

    Returns:
        分位值；空列表返回 0。
    """
    if not values:
        return 0.0
    return float(np.percentile(values, q))


@dataclass
class EvalRecord:
    """一条评测记录（逐样本保存，便于二次分析）。

    Attributes:
        item_id: 样本 id。
        category: 类别。
        answerable: 是否可答。
        abstained: 是否弃答。
        answer: 预测答案。
        gold_answer: 标准答案。
        confidence: 置信度。
        latency_ms: 延迟。
        contains: 答案包含标准值（0/1）。
        em: 完全匹配得分。
        f1: F1 得分。
        cite_hit: 引用是否命中。
        cite_precision: 引用精确率。
    """

    item_id: str
    category: str
    answerable: bool
    abstained: bool
    answer: str
    gold_answer: str
    confidence: float
    latency_ms: float
    contains: float = 0.0
    em: float = 0.0
    f1: float = 0.0
    cite_hit: bool = False
    cite_precision: float = 0.0


@dataclass
class MetricsSummary:
    """聚合指标。

    Attributes:
        n: 样本数。
        coverage: 答题率。
        accuracy: 可答已答样本上的 AnswerAcc。
        em: 可答已答样本上的 EM。
        f1: 可答已答样本上的 F1。
        cite_precision: 引用精确率。
        cite_recall: 引用召回率。
        abstain_accuracy: 弃答判断正确率。
        ece: 校准误差。
        latency_p50: 延迟中位数。
        latency_p95: 延迟 95 分位。
        by_category: 分类别 AnswerAcc。
    """

    n: int = 0
    coverage: float = 0.0
    accuracy: float = 0.0
    em: float = 0.0
    f1: float = 0.0
    cite_precision: float = 0.0
    cite_recall: float = 0.0
    abstain_accuracy: float = 0.0
    ece: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """序列化。

        Returns:
            字典。
        """
        return {
            "n": self.n,
            "coverage": self.coverage,
            "accuracy": self.accuracy,
            "em": self.em,
            "f1": self.f1,
            "cite_precision": self.cite_precision,
            "cite_recall": self.cite_recall,
            "abstain_accuracy": self.abstain_accuracy,
            "ece": self.ece,
            "latency_p50_ms": self.latency_p50,
            "latency_p95_ms": self.latency_p95,
            "by_category_accuracy": self.by_category,
        }


def summarize(records: list[EvalRecord]) -> MetricsSummary:
    """把逐样本记录聚合成指标。

    Args:
        records: 记录列表。

    Returns:
        MetricsSummary。
    """
    if not records:
        return MetricsSummary()

    answered = [record for record in records if not record.abstained]
    answered_answerable = [record for record in answered if record.answerable]

    summary = MetricsSummary(n=len(records))
    summary.coverage = len(answered) / len(records)
    summary.accuracy = _mean([record.contains for record in answered_answerable])
    summary.em = _mean([record.em for record in answered_answerable])
    summary.f1 = _mean([record.f1 for record in answered_answerable])
    summary.cite_precision = _mean([record.cite_precision for record in answered_answerable])
    summary.cite_recall = _mean([1.0 if record.cite_hit else 0.0 for record in answered_answerable])

    correct_flags = [
        ((not record.abstained) if record.answerable else record.abstained) for record in records
    ]
    summary.abstain_accuracy = _mean([1.0 if flag else 0.0 for flag in correct_flags])
    summary.ece = expected_calibration_error(
        [record.confidence for record in records],
        [record.contains >= 1.0 and not record.abstained for record in records],
    )
    latencies = [record.latency_ms for record in records]
    summary.latency_p50 = percentile(latencies, 50)
    summary.latency_p95 = percentile(latencies, 95)

    for category in sorted({record.category for record in records}):
        subset = [record for record in answered_answerable if record.category == category]
        summary.by_category[category] = _mean([record.contains for record in subset])
    return summary


def _mean(values: list[float]) -> float:
    """安全求均值。

    Args:
        values: 数值列表。

    Returns:
        均值；空列表返回 0。
    """
    return float(sum(values) / len(values)) if values else 0.0


def safe_log(value: float) -> float:
    """防止 log(0) 的辅助函数（评测中偶尔需要）。

    Args:
        value: 输入。

    Returns:
        ln(max(value, 1e-12))。
    """
    return math.log(max(value, 1e-12))
