"""评测入口：一句话命令跑完"建索引 → 提问 → 打分 → 出报告"。

用法：
    uv run python -m visioncite.eval.run --mode all
    uv run python -m visioncite.eval.run --mode layout --per 2

三种模式（消融）：
    page    : 整页当一块（没有版面切分）—— 基线
    layout  : XY-cut 区域 + BM25 —— 只用文本结构
    hybrid  : 区域 + BM25 + CLIP 视觉召回 —— 完整系统

报告写到 ``reports/``，包含逐样本记录，方便你复核每个数字的来源。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from visioncite.answer.qa import answer_question
from visioncite.config import Settings
from visioncite.errors import VisionCiteError
from visioncite.eval.bootstrap import bootstrap_mean_ci
from visioncite.eval.corpus_gen import load_or_generate
from visioncite.eval.dataset import QAItem
from visioncite.eval.metrics import (
    EvalRecord,
    answer_contains,
    citation_hit,
    citation_precision,
    exact_match,
    summarize,
    token_f1,
)
from visioncite.index.chunking import chunks_from_pages
from visioncite.index.visual import NullVisualEncoder, create_visual
from visioncite.logging import get_logger, setup_logging
from visioncite.ocr.factory import create_ocr
from visioncite.pipeline import Document, build_document
from visioncite.types import BBox, Region

logger = get_logger("eval.run")
MODES = ("page", "layout", "hybrid")


def build_documents(
    corpus_root: Path,
    images: list[Path],
    mode: str,
    settings: Settings,
) -> dict[str, Document]:
    """为语料里每张图建索引。

    Args:
        corpus_root: 语料根目录。
        images: 图片路径列表。
        mode: 评测模式。
        settings: 配置。

    Returns:
        相对路径 → Document。
    """
    visual = create_visual(settings) if mode == "hybrid" else NullVisualEncoder()
    ocr = create_ocr(settings)
    documents: dict[str, Document] = {}
    for image in images:
        relative = image.relative_to(corpus_root).as_posix()
        started = time.perf_counter()
        document = build_document(image, settings, ocr=ocr, visual=visual)
        if mode == "page":
            _flatten_to_page(document, settings)
        documents[relative] = document
        logger.info("%s 建索引 %.1fs", relative, time.perf_counter() - started)
    return documents


def evaluate(
    documents: dict[str, Document],
    items: list[QAItem],
    settings: Settings,
) -> list[EvalRecord]:
    """逐样本问答并打分。

    Args:
        documents: 文档索引。
        items: 样本。
        settings: 配置。

    Returns:
        记录列表。
    """
    records: list[EvalRecord] = []
    for item in items:
        document = documents[item.doc_file]
        result = answer_question(document, item.question, settings)
        answered = not result.abstained
        scored = answered and item.answerable
        records.append(
            EvalRecord(
                item_id=item.item_id,
                category=item.category,
                answerable=item.answerable,
                abstained=result.abstained,
                answer=result.answer,
                gold_answer=item.answer,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                contains=answer_contains(result.answer, item.answer) if scored else 0.0,
                em=exact_match(result.answer, item.answer) if scored else 0.0,
                f1=token_f1(result.answer, item.answer) if scored else 0.0,
                cite_hit=citation_hit(list(result.citations), item.citation) if answered else False,
                cite_precision=(
                    citation_precision(list(result.citations), item.citation) if answered else 0.0
                ),
            )
        )
    return records


def run_mode(
    mode: str,
    corpus_root: Path,
    images: list[Path],
    items: list[QAItem],
    settings: Settings,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """跑一个模式并返回汇总与逐样本记录。

    Args:
        mode: 模式名。
        corpus_root: 语料根目录。
        images: 图片列表。
        items: 样本。
        settings: 配置。

    Returns:
        ``(汇总字典, 记录列表)``。
    """
    documents = build_documents(corpus_root, images, mode, settings)
    records = evaluate(documents, items, settings)
    summary = summarize(records)
    answered_answerable = [r for r in records if r.answerable and not r.abstained]
    low, high = bootstrap_mean_ci([r.contains for r in answered_answerable])
    payload = {
        "mode": mode,
        "summary": summary.to_dict(),
        "accuracy_ci95": [low, high],
        "records": len(records),
    }
    return payload, [asdict(record) | {"mode": mode} for record in records]


def write_report(results: list[dict[str, object]], out_dir: Path, corpus_info: dict[str, object]) -> Path:
    """写 Markdown 报告。

    Args:
        results: 各模式的结果。
        out_dir: 输出目录。
        corpus_info: 语料信息。

    Returns:
        报告路径。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"eval-{stamp}.md"
    lines = [
        "# VisionCite 评测报告",
        "",
        f"- 生成时间：{stamp}",
        f"- 语料：{corpus_info['total']} 个问题 / {corpus_info['images']} 张图",
        f"- 可答题：{corpus_info['answerable']}，不可答：{corpus_info['unanswerable']}",
        "- 指标口径见 docs/EVALUATION.md（AnswerAcc 只统计可答且实际作答的样本）",
        "",
        "| 模式 | 答题率 | AnswerAcc | 95% CI | F1 | Cite-P | Cite-R | 弃答准确 | ECE | p50 | p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        summary = result["summary"]
        ci = result["accuracy_ci95"]
        lines.append(
            f"| {result['mode']} | {summary['coverage']:.2f} | {summary['accuracy']:.2f} | "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] | {summary['f1']:.2f} | {summary['cite_precision']:.2f} | "
            f"{summary['cite_recall']:.2f} | {summary['abstain_accuracy']:.2f} | {summary['ece']:.2f} | "
            f"{summary['latency_p50_ms']:.0f} ms | {summary['latency_p95_ms']:.0f} ms |"
        )
    lines.extend(
        ["", "## 分类别 AnswerAcc", "", "| 模式 | invoice | table | chart | slide |", "|---|---|---|---|---|"]
    )
    for result in results:
        by = result["summary"]["by_category_accuracy"]
        lines.append(
            f"| {result['mode']} | {by.get('invoice', 0):.2f} | {by.get('table', 0):.2f} | "
            f"{by.get('chart', 0):.2f} | {by.get('slide', 0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 语料为程序合成，标准答案与证据框由渲染坐标精确得到；不代表真实扫描件的难度。",
            "- ECE 的\"正确\"定义为 AnswerAcc=1 且未弃答。",
            "- 报告由 `python -m visioncite.eval.run` 自动生成，数字未经人工修改。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _flatten_to_page(document: Document, settings: Settings) -> None:
    """消融用：把每页压成"整页一个区域"，去掉版面结构。

    Args:
        document: 文档（原地修改）。
        settings: 配置（BM25 参数）。
    """
    for page in document.pages:
        text = " ".join(word.text for region in page.regions for word in region.words)
        page.regions = [
            Region(
                region_id=f"p{page.index}-r0",
                page=page.index,
                bbox=BBox(0, 0, page.width, page.height),
                kind="text",
                text=text or page.text_layer,
                confidence=1.0,
            )
        ]
    document.chunks = chunks_from_pages(document.meta.doc_id, document.pages)
    document.bm25.fit([(chunk.chunk_id, chunk.text) for chunk in document.chunks])
    document.vectors = None


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表；None 用 ``sys.argv``。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="VisionCite 评测")
    parser.add_argument("--corpus", default="corpus", help="语料目录")
    parser.add_argument("--out", default="reports", help="报告输出目录")
    parser.add_argument("--mode", default="all", choices=("all", *MODES))
    parser.add_argument("--per", type=int, default=2, help="每类模板生成几份")
    args = parser.parse_args(argv)

    setup_logging()
    settings = Settings.from_env()
    settings.ensure_dirs()
    corpus_root = Path(args.corpus)
    try:
        corpus = load_or_generate(corpus_root, per_template=args.per)
    except VisionCiteError as exc:
        logger.error("语料生成失败: %s", exc.user_message)
        return 2

    corpus_info = {
        "total": len(corpus.items),
        "images": len(corpus.images),
        "answerable": sum(1 for item in corpus.items if item.answerable),
        "unanswerable": sum(1 for item in corpus.items if not item.answerable),
    }
    modes = MODES if args.mode == "all" else (args.mode,)

    results: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []
    for mode in modes:
        logger.info("开始评测模式 %s", mode)
        payload, rows = run_mode(mode, corpus_root, corpus.images, corpus.items, settings)
        results.append(payload)
        all_records.extend(rows)

    report_path = write_report(results, Path(args.out), corpus_info)
    json_path = report_path.with_suffix(".json")
    payload = {"corpus": corpus_info, "results": results, "records": all_records}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: {report_path}")
    for result in results:
        summary = result["summary"]
        print(
            f"{result['mode']:7s} acc={summary['accuracy']:.3f} "
            f"coverage={summary['coverage']:.2f} f1={summary['f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
