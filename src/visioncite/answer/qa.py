"""问答编排：检索 → 作答 → 校验 → 弃答。

这是整个系统的"用例层"：它只依赖接口（BM25、视觉编码器、答案器、校验器），
因此每一环都能单独替换，也方便测试时注入假实现。

流程与失败策略：
    检索为空          → 弃答
    extractive 没候选 → 弃答
    VLM 失败/超时     → 回退 extractive（记日志，不向用户报错）
    VLM 引用无效      → 回退 extractive
    校验不通过        → 弃答（并给出原因）
    置信度不足        → 弃答
"""

from __future__ import annotations

import time

from visioncite.answer.abstain import should_abstain
from visioncite.answer.extractive import compose_answer
from visioncite.answer.verify import verify_answer
from visioncite.answer.vlm import VlmAnswerer, VlmOutput
from visioncite.config import Settings
from visioncite.index.retrieve import Retrieved, retrieve
from visioncite.ingest.image import crop_image, to_png_bytes
from visioncite.logging import get_logger
from visioncite.pipeline import Document
from visioncite.types import AnswerResult, Chunk, Citation

logger = get_logger("answer.qa")


def answer_question(
    document: Document,
    question: str,
    settings: Settings,
    backend: str = "auto",
) -> AnswerResult:
    """回答问题并给出证据。

    Args:
        document: 已建索引的文档。
        question: 用户问题。
        settings: 配置。
        backend: ``auto``（有 VLM 就用）、``extractive`` 或 ``vlm``。

    Returns:
        完整结果；任何异常路径都体现在 ``abstained`` 与 ``reasons`` 中。
    """
    started = time.perf_counter()
    question = question.strip()
    if not question:
        return _abstain("问题为空", started)

    results = retrieve(document, question, settings)
    if not results:
        return _abstain("没有检索到相关证据", started)

    chunks = [item.chunk for item in results]
    answer_text, cited, confidence, backend_used = _compose(
        document, question, chunks, settings, backend
    )

    citations = _build_citations(cited, results, settings)
    verification = verify_answer(answer_text, citations, chunks)
    abstained, reasons = should_abstain(confidence, len(citations), verification, settings)
    if abstained:
        answer_text = ""

    return AnswerResult(
        answer=answer_text,
        citations=tuple(citations),
        abstained=abstained,
        confidence=confidence,
        latency_ms=(time.perf_counter() - started) * 1000,
        backend=backend_used,
        reasons=reasons,
    )


def _compose(
    document: Document,
    question: str,
    chunks: list[Chunk],
    settings: Settings,
    backend: str,
) -> tuple[str, list[Chunk], float, str]:
    """选择作答后端并生成答案。

    Args:
        document: 文档（VLM 需要页面图）。
        question: 问题。
        chunks: 证据块。
        settings: 配置。
        backend: 后端偏好。

    Returns:
        ``(答案, 引用块, 置信度, 实际后端名)``。
    """
    if settings.vlm_enabled and backend in {"auto", "vlm"}:
        try:
            output = _call_vlm(document, question, chunks, settings)
            cited = _map_citations(output.citations, chunks)
            if cited and output.answer:
                return output.answer, cited, output.confidence, "vlm"
            logger.info("VLM 未给出可用答案，回退 extractive")
        except Exception:  # noqa: BLE001 - 外部依赖失败必须能降级
            logger.exception("VLM 调用失败，回退 extractive")

    answer, cited, confidence = compose_answer(question, chunks)
    return answer, cited, confidence, "extractive"


def _call_vlm(
    document: Document,
    question: str,
    chunks: list[Chunk],
    settings: Settings,
) -> VlmOutput:
    """把证据裁成图，交给 VLM。

    Args:
        document: 文档。
        question: 问题。
        chunks: 证据块。
        settings: 配置（VLM 地址与模型名）。

    Returns:
        VlmOutput。
    """
    crops = {}
    for chunk in chunks:
        page = document.page(chunk.page)
        if page is None:
            continue
        region = chunk.bbox.pad(8, (page.width, page.height))
        crops[chunk.chunk_id] = to_png_bytes(crop_image(page.image, region))

    client = VlmAnswerer(settings.vlm_base_url, settings.vlm_model)
    return client.answer(question, chunks, crops)


def _map_citations(ids: tuple[str, ...], chunks: list[Chunk]) -> list[Chunk]:
    """把模型返回的 id 映射成块，未知 id 直接丢弃（引用白名单）。

    Args:
        ids: 模型给出的 chunk_id。
        chunks: 允许引用的块。

    Returns:
        有效块列表，保持模型给出的顺序。
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]


def _build_citations(cited: list[Chunk], results: list[Retrieved], settings: Settings) -> list[Citation]:
    """把块转成引用，分数取自检索结果。

    低于 ``settings.min_citation_score`` 的引用会被丢弃：
    它们通常是"顺带召回"的弱证据，展示出来只会干扰用户判断。

    Args:
        cited: 被引块。
        results: 检索结果。
        settings: 配置。

    Returns:
        引用列表，保持顺序。
    """
    score_by_id = {item.chunk.chunk_id: item.score for item in results}
    citations: list[Citation] = []
    for chunk in cited:
        score = score_by_id.get(chunk.chunk_id, 0.0)
        if score < settings.min_citation_score:
            continue
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                region_id=chunk.region_id,
                page=chunk.page,
                bbox=chunk.bbox,
                snippet=chunk.text[:160],
                score=score,
            )
        )
    return citations


def _abstain(reason: str, started: float) -> AnswerResult:
    """构造弃答结果。

    Args:
        reason: 原因。
        started: 起始时间戳（perf_counter）。

    Returns:
        AnswerResult。
    """
    return AnswerResult(
        answer="",
        abstained=True,
        confidence=0.0,
        latency_ms=(time.perf_counter() - started) * 1000,
        reasons=(reason,),
    )
