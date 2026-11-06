"""作答层测试：抽取、校验、弃答。"""

from __future__ import annotations

from visioncite.answer.abstain import should_abstain
from visioncite.answer.extractive import compose_answer, query_terms, score_sentence
from visioncite.answer.verify import (
    citation_ids_exist,
    lexical_support,
    numbers_in,
    numbers_supported,
    verify_answer,
)
from visioncite.config import Settings
from visioncite.types import BBox, Chunk, Citation


def _chunk(chunk_id: str, text: str) -> Chunk:
    """构造一个测试用块。"""
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page=0,
        region_id=chunk_id.split(":")[1],
        bbox=BBox(0, 0, 100, 20),
        kind="text",
        text=text,
    )


def _citation(chunk: Chunk) -> Citation:
    """构造一条指向某块的引用（测试里反复用到）。"""
    return Citation(
        chunk_id=chunk.chunk_id,
        region_id=chunk.region_id,
        page=chunk.page,
        bbox=BBox(0, 0, 1, 1),
        snippet="x",
    )


def test_query_terms_drop_question_words() -> None:
    """"是多少/什么"这类疑问字符不应主导覆盖率。"""
    assert "合计" in query_terms("合计金额是多少？")
    assert "多少" not in query_terms("合计金额是多少？")


def test_score_sentence_rewards_matching_terms() -> None:
    """命中查询词越多，分数越高。"""
    high, _ = score_sentence("合计金额：1286.40 元", ["合计", "金额"])
    low, _ = score_sentence("开票日期：2026-10-25", ["合计", "金额"])
    assert high > low


def test_compose_answer_returns_evidence_sentence() -> None:
    """答案应来自证据句，且引用指向该块。"""
    chunk = _chunk("doc:p0-r1:0", "合计金额：1286.40 元。开票日期：2026-10-25。")
    answer, cited, confidence = compose_answer("合计金额是多少？", [chunk])
    assert "1286.40" in answer
    assert cited[0].chunk_id == chunk.chunk_id
    assert 0 < confidence <= 1


def test_compose_answer_without_evidence() -> None:
    """没有证据时返回空答案与 0 置信度。"""
    assert compose_answer("金额是多少？", []) == ("", [], 0.0)


def test_numbers_in_handles_currency_and_thousands() -> None:
    """货币符号与千分位要被归一化。"""
    assert numbers_in("¥1,286.40") == ["1286.40"]


def test_numbers_supported_true() -> None:
    """数字出现在证据里算通过。"""
    assert numbers_supported("合计 1286.40 元", ["合计金额：1,286.40 元"])


def test_numbers_supported_false() -> None:
    """数字不在证据里必须判失败。"""
    assert not numbers_supported("合计 999.00 元", ["合计金额：1,286.40 元"])


def test_lexical_support_ratio() -> None:
    """词面支撑度在 0~1 之间。"""
    assert lexical_support("合计金额", ["合计金额：1286.40 元"]) == 1.0


def test_citation_ids_exist() -> None:
    """伪造的 chunk_id 要被抓出来。"""
    chunk = _chunk("doc:p0-r1:0", "金额")
    good = Citation(chunk_id=chunk.chunk_id, region_id="p0-r1", page=0, bbox=BBox(0, 0, 1, 1), snippet="金额")
    bad = Citation(chunk_id="不存在", region_id="p0-r9", page=0, bbox=BBox(0, 0, 1, 1), snippet="金额")
    assert citation_ids_exist([good], [chunk])
    assert not citation_ids_exist([bad], [chunk])


def test_verify_answer_passes_for_grounded_answer() -> None:
    """答案与引用一致时校验通过。"""
    chunk = _chunk("doc:p0-r1:0", "合计金额：1286.40 元")
    citation = _citation(chunk)
    result = verify_answer("合计金额：1,286.40 元", [citation], [chunk])
    assert result.ok


def test_verify_answer_fails_on_fabricated_number() -> None:
    """答案数字与证据不符时校验失败。"""
    chunk = _chunk("doc:p0-r1:0", "合计金额：1286.40 元")
    citation = _citation(chunk)
    result = verify_answer("合计金额：9999.00 元", [citation], [chunk])
    assert not result.ok
    assert any("数字" in reason for reason in result.reasons)


def test_abstain_when_no_citations() -> None:
    """没有引用时必须弃答。"""
    settings = Settings()
    ok = verify_answer("", [], [])
    abstained, reasons = should_abstain(0.9, 0, ok, settings)
    assert abstained and reasons


def test_abstain_when_low_confidence() -> None:
    """置信度低于阈值时弃答。"""
    settings = Settings(abstain_threshold=0.5)
    ok = verify_answer("", [], [])
    abstained, _ = should_abstain(0.2, 1, ok, settings)
    assert abstained


def test_no_abstain_when_confident_and_verified() -> None:
    """置信度足够且校验通过时不弃答。"""
    settings = Settings(abstain_threshold=0.3)
    chunk = _chunk("doc:p0-r1:0", "合计金额：1286.40 元")
    citation = _citation(chunk)
    ok = verify_answer("合计金额：1,286.40 元", [citation], [chunk])
    abstained, reasons = should_abstain(0.8, 1, ok, settings)
    assert not abstained and not reasons
