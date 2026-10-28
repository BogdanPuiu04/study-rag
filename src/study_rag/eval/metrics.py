"""Retrieval metrics. Relevance is judged per chunk with a golden label of
(source file, phrase the chunk must contain). Labelling by phrase rather than
chunk_id keeps the golden set valid when chunking parameters change."""

from __future__ import annotations

from dataclasses import dataclass

from study_rag.models import ScoredChunk


@dataclass(frozen=True)
class GoldenItem:
    question: str
    source: str
    must_contain: str


def _normalize(text: str) -> str:
    # PDF extraction breaks lines mid-sentence, so a phrase copied from a PDF viewer
    # ("vanishing gradient") may be "vanishing\ngradient" in the chunk. Compare with
    # whitespace collapsed, otherwise real hits count as misses.
    return " ".join(text.lower().split())


def is_relevant(sc: ScoredChunk, item: GoldenItem) -> bool:
    c = sc.chunk
    return c.source == item.source and _normalize(item.must_contain) in _normalize(c.text)


def first_relevant_rank(results: list[ScoredChunk], item: GoldenItem) -> int | None:
    """1-based position of the first relevant chunk in `results`, or None if absent."""
    for rank, sc in enumerate(results, start=1):
        if is_relevant(sc, item):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose relevant chunk ranked within the top k.

    `ranks` holds one entry per question (output of first_relevant_rank).
    Empty input -> 0.0.
    """
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank: average of 1/rank, counting a miss (None) as 0.

    Recall@k only asks "was it in the top k?"; MRR also rewards putting it first. Two systems
    with the same recall@5 can differ a lot here, and the top position matters: the model
    weighs early excerpts more, and a good rank 1 lets `top_k` (and the token bill) shrink.
    """
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None) / len(ranks)
