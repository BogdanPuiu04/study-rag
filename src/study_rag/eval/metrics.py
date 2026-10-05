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


def is_relevant(sc: ScoredChunk, item: GoldenItem) -> bool:
    c = sc.chunk
    return c.source == item.source and item.must_contain.lower() in c.text.lower()


# --- EXERCISE 1 ---------------------------------------------------------------


def first_relevant_rank(results: list[ScoredChunk], item: GoldenItem) -> int | None:
    """1-based position of the first relevant chunk in `results`, or None if absent."""
    raise NotImplementedError("exercise 1")


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose relevant chunk ranked within the top k.

    `ranks` holds one entry per question (output of first_relevant_rank).
    Empty input -> 0.0.
    """
    raise NotImplementedError("exercise 1")


def mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank: average of 1/rank, counting a miss (None) as 0.

    Think about it: why does MRR tell you something recall@5 doesn't?
    """
    raise NotImplementedError("exercise 1")
