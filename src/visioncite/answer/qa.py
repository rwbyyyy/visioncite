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
from visioncite.answer.color import dominant_background, dominant_color, find_red_bbox
from visioncite.answer.extractive import compose_answer
from visioncite.answer.values import asks_for_value, extract_value, split_cells
from visioncite.answer.verify import verify_answer
from visioncite.answer.vlm import VlmAnswerer, VlmOutput
from visioncite.config import Settings
from visioncite.index.retrieve import Retrieved, retrieve
from visioncite.ingest.image import crop_image, to_png_bytes
from visioncite.logging import get_logger
from visioncite.pipeline import Document
from visioncite.types import AnswerResult, BBox, Chunk, Citation

logger = get_logger("answer.qa")

# 颜色类问题的关键词：命中后走像素统计路径，而不是文本抽取
_COLOR_KEYWORDS = ("颜色", "什么色", "字体色", "底色", "色调", "色号")


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

    # 颜色属于"视觉属性"，OCR 文本里没有答案，交给像素统计
    if _is_color_question(question):
        color_result = _answer_color(document, question, settings, started)
        if color_result is not None:
            return color_result

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


def _asks_background(question: str) -> bool:
    """问题是否在问底色/背景色。

    Args:
        question: 用户问题。

    Returns:
        True 表示应统计背景像素。
    """
    return any(keyword in question for keyword in ("底色", "背景色", "纸张颜色"))


def _is_color_question(question: str) -> bool:
    """判断问题是否在问颜色。

    Args:
        question: 用户问题。

    Returns:
        True 表示应按视觉属性回答。
    """
    return any(keyword in question for keyword in _COLOR_KEYWORDS)


def _answer_color(
    document: Document,
    question: str,
    settings: Settings,
    started: float,
) -> AnswerResult | None:
    """用像素统计回答颜色问题。

    先用 BM25 找到"问的是哪一块"（例如"成绩报告单"对应的那行），
    再统计这块的墨迹主色；找不到具体区域就退回整页统计。

    Args:
        document: 文档。
        question: 问题。
        settings: 配置。
        started: 起始时间戳。

    Returns:
        AnswerResult；拿不到有效统计时返回 None（交回文本路径处理）。
    """
    if not document.pages:
        return None

    results = retrieve(document, question, settings)
    page_index = 0
    region = None
    if results:
        chunk = results[0].chunk
        page_index = chunk.page
        page = document.page(page_index)
        if page is not None:
            region = next((item for item in page.regions if item.region_id == chunk.region_id), None)

    page = document.page(page_index)
    if page is None:
        return None

    # 问印章时，先定位红色区域；印章是整页里唯一的红块，比"整页主色"准
    if any(keyword in question for keyword in ("印章", "公章", "盖章", "红章")):
        seal_box = find_red_bbox(page.image)
        if seal_box is not None:
            stat = dominant_color(page.image, seal_box)
            if stat is not None:
                citation = Citation(
                    chunk_id=f"{document.meta.doc_id}:color:seal",
                    region_id=f"p{page_index}-seal",
                    page=page_index,
                    bbox=seal_box,
                    snippet=f"印章区域主色：{stat.name}",
                    score=1.0,
                )
                return AnswerResult(
                    answer=f"印章区域是{stat.name}（占该区域墨迹像素 {stat.ratio:.0%}）",
                    citations=(citation,),
                    abstained=False,
                    confidence=min(1.0, 0.6 + 0.4 * min(1.0, stat.ratio / 0.8)),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    backend="color",
                )

    # 问"底色"通常指整张纸，而不是某行字所在的区域
    if _asks_background(question):
        region = None

    stat = (
        dominant_background(page.image, region.bbox if region else None)
        if _asks_background(question)
        else dominant_color(page.image, region.bbox if region else None)
    )
    if stat is None:
        return _abstain("目标区域没有可统计的像素", started)

    subject = f"「{region.text[:24]}」" if region and region.text else "整页"
    kind = "底色" if _asks_background(question) else "主色"
    answer = f"{subject}的{kind}是{stat.name}（占该区域像素 {stat.ratio:.0%}）"
    bbox = region.bbox if region else BBox(0, 0, page.width, page.height)
    citation = Citation(
        chunk_id=f"{document.meta.doc_id}:color:{region.region_id if region else 'page'}",
        region_id=region.region_id if region else f"p{page_index}-r0",
        page=page_index,
        bbox=bbox,
        snippet=answer,
        score=1.0,
    )
    confidence = min(1.0, 0.55 + 0.45 * min(1.0, stat.ratio / 0.8))
    return AnswerResult(
        answer=answer,
        citations=(citation,),
        abstained=False,
        confidence=confidence,
        latency_ms=(time.perf_counter() - started) * 1000,
        backend="color",
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
    answer, confidence = _refine_value(document, question, answer, cited, confidence)
    return answer, cited, confidence, "extractive"


def _refine_value(
    document: Document,
    question: str,
    answer: str,
    cited: list[Chunk],
    confidence: float,
) -> tuple[str, float]:
    """如果用户在问具体数值，把整行答案收敛成那个值。

    Args:
        document: 文档（用于找同区域的表头）。
        question: 用户问题。
        answer: 抽取式答案。
        cited: 引用的块。
        confidence: 原置信度。

    Returns:
        ``(收敛后的答案, 调整后的置信度)``。
    """
    if not asks_for_value(question) or not answer:
        return answer, confidence
    header = _header_for(document, cited[0]) if cited else None
    value = extract_value(question, answer, header)
    if not value or value == answer:
        return answer, confidence
    return value, min(1.0, confidence + 0.1)


def _header_for(document: Document, chunk: Chunk) -> str | None:
    """找同一表格区域的表头行（区域里的第一块）。

    Args:
        document: 文档。
        chunk: 数据行块。

    Returns:
        表头文本；找不到返回 None。
    """
    prefix = f"{document.meta.doc_id}:{chunk.region_id}:"
    for candidate in document.chunks:
        if candidate.chunk_id == chunk.chunk_id:
            continue
        if candidate.chunk_id.startswith(prefix) and candidate.chunk_id.endswith(":0"):
            return candidate.text
    return _find_table_header(document, chunk)


def _find_table_header(document: Document, chunk: Chunk) -> str | None:
    """跨区域找表头：表格每一行可能被切成独立区域，表头不一定同区域。

    判据：与数据行**列数相同**、且几乎不含数字的行，视为表头。

    Args:
        document: 文档。
        chunk: 数据行块。

    Returns:
        表头文本；找不到返回 None。
    """
    columns = len([cell for cell in split_cells(chunk.text) if cell])
    if columns < 2:
        return None
    for candidate in document.chunks:
        if candidate.chunk_id == chunk.chunk_id:
            continue
        cells = split_cells(candidate.text)
        if len(cells) != columns:
            continue
        numeric = sum(1 for cell in cells if cell and cell.replace(".", "").isdigit())
        if numeric <= 1 and any(cell and not cell.replace(".", "").isdigit() for cell in cells):
            return candidate.text
    return None


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
