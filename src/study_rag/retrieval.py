"""Hybrid retrieval: vector + BM25, fused with Reciprocal Rank Fusion, then optional rerank.

Why hybrid? Embeddings catch paraphrases ("how does the model learn" -> backprop),
keyword search catches exact terms embeddings blur (course codes, formula names,
acronyms like "SVM"). RRF combines ranks rather than raw scores, so we don't have
to calibrate cosine similarity against BM25 — they live on different scales.
"""

from __future__ import annotations

from typing import Protocol

from study_rag.embeddings import Embedder
from study_rag.models import ScoredChunk
from study_rag.store.base import VectorStore


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """EXERCISE 4. Merge several rankings into one.
    Each chunk scores sum(1 / (k + rank)) over the rankings it appears in
    (rank is 1-based). Return the unique chunks sorted by that score, highest
    first, with the fused score in ScoredChunk.score. Identify chunks by chunk_id.

    Why ranks and not the raw scores? Compare the range of a cosine score
    with the range of a BM25 score.
    """

    raise NotImplementedError("exercise 4")


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly — slower than bi-encoders, much more precise.
    Only applied to the fused top-N candidates, so cost stays bounded."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """EXERCISE 6. Re-order candidates by cross-encoder relevance, highest first.

        self.model.predict(list_of_(query, passage)_pairs) returns one score per pair.
        Return new ScoredChunks carrying the cross-encoder score. Handle [] input.
        """
        raise NotImplementedError("exercise 6")


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        mode: str = "hybrid",
    ) -> None:
        if mode not in {"hybrid", "vector", "keyword"}:
            raise ValueError(f"unknown mode: {mode}")
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.candidate_k, self.mode = candidate_k, mode

    def retrieve(self, query: str, k: int = 5, course: str | None = None) -> list[ScoredChunk]:
        # With a course filter, over-fetch so filtering still leaves enough candidates.
        n = self.candidate_k * (4 if course else 1)
        rankings = []
        if self.mode in {"hybrid", "vector"}:
            qvec = self.embedder.embed([query])[0]
            rankings.append(self.store.vector_search(qvec, n))
        if self.mode in {"hybrid", "keyword"}:
            rankings.append(self.store.keyword_search(query, n))
        fused = reciprocal_rank_fusion(rankings)
        if course:
            fused = [sc for sc in fused if course_of(sc) == course]
        fused = fused[: self.candidate_k]
        if self.reranker:
            fused = self.reranker.rerank(query, fused)
        return fused[:k]


def course_of(sc: ScoredChunk) -> str:
    """Course = first folder under data/, e.g. "machine-learning/week1.md" -> "machine-learning"."""
    parts = sc.chunk.source.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else ""
