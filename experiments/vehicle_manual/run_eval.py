"""在"车辆用户手册"语料上跑 VisionCite 的 RAG 评测。

输出：reports/vehicle-manual-<时间>.md + .json（逐样本记录可复核）。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from visioncite.answer.qa import answer_question
from visioncite.config import Settings
from visioncite.eval.bootstrap import bootstrap_mean_ci
from visioncite.eval.metrics import (
    EvalRecord,
    answer_contains,
    citation_hit,
    citation_precision,
    exact_match,
    summarize,
    token_f1,
)
from visioncite.index.visual import NullVisualEncoder, create_visual
from visioncite.logging import get_logger, setup_logging
from visioncite.pipeline import Document, build_document

ROOT = Path(__file__).resolve().parent
logger = get_logger("vehicle-manual")


def load_items() -> list[dict]:
    """读取问答清单。"""
    lines = (ROOT / "items.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_documents(use_visual: bool, settings: Settings) -> dict[str, Document]:
    """为每页建索引。"""
    visual = create_visual(settings) if use_visual else NullVisualEncoder()
    documents: dict[str, Document] = {}
    for image in sorted((ROOT / "images").glob("*.png")):
        relative = image.relative_to(ROOT).as_posix()
        started = time.perf_counter()
        documents[relative] = build_document(image, settings, visual=visual)
        logger.info("%s 索引完成 %.1fs", relative, time.perf_counter() - started)
    return documents


def evaluate(documents: dict[str, Document], items: list[dict], settings: Settings) -> list[EvalRecord]:
    """逐题问答并打分。"""
    records: list[EvalRecord] = []
    for item in items:
        document = documents[item["file"]]
        result = answer_question(document, item["question"], settings)
        answered = not result.abstained
        gold_bbox = None
        if item.get("citation"):
            from visioncite.types import BBox

            gold_bbox = BBox.from_list(item["citation"])
        records.append(
            EvalRecord(
                item_id=item["id"],
                category=item["category"],
                answerable=item["answerable"],
                abstained=result.abstained,
                answer=result.answer,
                gold_answer=item["answer"],
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                contains=(
                    answer_contains(result.answer, item["answer"])
                    if answered and item["answerable"]
                    else 0.0
                ),
                em=(
                    exact_match(result.answer, item["answer"])
                    if answered and item["answerable"]
                    else 0.0
                ),
                f1=(
                    token_f1(result.answer, item["answer"])
                    if answered and item["answerable"]
                    else 0.0
                ),
                cite_hit=citation_hit(list(result.citations), gold_bbox) if answered else False,
                cite_precision=(
                    citation_precision(list(result.citations), gold_bbox) if answered else 0.0
                ),
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="启用 CLIP 视觉召回")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    setup_logging()
    settings = Settings.from_env()
    settings.ensure_dirs()

    items = load_items()
    documents = build_documents(args.visual, settings)
    records = evaluate(documents, items, settings)
    summary = summarize(records)
    answered = [r for r in records if r.answerable and not r.abstained]
    low, high = bootstrap_mean_ci([r.contains for r in answered])

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "mode": "visual" if args.visual else "text",
        "items": len(items),
        "answered_answerable": len(answered),
        "summary": summary.to_dict(),
        "accuracy_ci95": [low, high],
        "records": [asdict(record) for record in records],
    }
    (out_dir / f"vehicle-manual-{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 车辆用户手册 RAG 评测",
        "",
        f"- 时间：{stamp}",
        f"- 模式：{'BM25 + CLIP' if args.visual else 'BM25（文本）'}",
        f"- 题量：{len(items)}（可答 {sum(1 for i in items if i['answerable'])}）",
        "",
        "| 答题率 | AnswerAcc | 95% CI | F1 | Cite-P | Cite-R | 弃答准确率 | ECE | p50 | p95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
        (
            f"| {summary.coverage:.2f} | {summary.accuracy:.2f} | [{low:.2f}, {high:.2f}] | "
            f"{summary.f1:.2f} | {summary.cite_precision:.2f} | {summary.cite_recall:.2f} | "
            f"{summary.abstain_accuracy:.2f} | {summary.ece:.2f} | "
            f"{summary.latency_p50:.1f} ms | {summary.latency_p95:.1f} ms |"
        ),
        "",
        "## 失败样本",
        "",
    ]
    for record in records:
        failed = (record.answerable and (record.abstained or record.contains < 1)) or (
            not record.answerable and not record.abstained
        )
        if failed:
            lines.append(
                f"- `{record.item_id}` gold={record.gold_answer!r} pred={record.answer!r} "
                f"abstained={record.abstained}"
            )
    report = out_dir / f"vehicle-manual-{stamp}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {report}")
    print(
        f"coverage={summary.coverage:.2f} acc={summary.accuracy:.2f} "
        f"cite_p={summary.cite_precision:.2f} cite_r={summary.cite_recall:.2f} "
        f"abstain={summary.abstain_accuracy:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
