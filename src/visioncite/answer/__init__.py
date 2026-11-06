"""作答层：抽取式答案、可选 VLM、答案校验、弃答策略。"""

from visioncite.answer.abstain import should_abstain
from visioncite.answer.extractive import compose_answer, query_terms, rank_candidates
from visioncite.answer.qa import answer_question
from visioncite.answer.verify import (
    Verification,
    lexical_support,
    numbers_in,
    numbers_supported,
    verify_answer,
)
from visioncite.answer.vlm import VlmAnswerer, VlmOutput

__all__ = [
    "Verification",
    "VlmAnswerer",
    "VlmOutput",
    "answer_question",
    "compose_answer",
    "lexical_support",
    "numbers_in",
    "numbers_supported",
    "query_terms",
    "rank_candidates",
    "should_abstain",
    "verify_answer",
]
