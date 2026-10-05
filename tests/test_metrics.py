"""Exercise 1."""

import pytest

from study_rag.eval.metrics import GoldenItem, first_relevant_rank, mrr, recall_at_k
from study_rag.models import Chunk, ScoredChunk

ITEM = GoldenItem("q", source="a.md", must_contain="chain rule")


def sc(source: str, text: str) -> ScoredChunk:
    return ScoredChunk(Chunk("id", "d", text, source), 0.0)


def test_first_relevant_rank_is_one_based():
    results = [sc("a.md", "nothing"), sc("b.md", "chain rule"), sc("a.md", "the Chain Rule")]
    assert first_relevant_rank(results, ITEM) == 3


def test_first_relevant_rank_none_when_missing():
    assert first_relevant_rank([sc("a.md", "nope")], ITEM) is None


def test_recall_at_k():
    assert recall_at_k([1, 3, None, 6], k=5) == pytest.approx(0.5)
    assert recall_at_k([], k=5) == 0.0


def test_mrr_counts_misses_as_zero():
    assert mrr([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert mrr([]) == 0.0
