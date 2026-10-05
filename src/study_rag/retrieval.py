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
        if not candidates:
            return []
        scores = self.model.predict([(query, c.chunk.text) for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(c.chunk, float(s)) for c, s in ranked]


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

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        rankings = []
        if self.mode in {"hybrid", "vector"}:
            qvec = self.embedder.embed([query])[0]
            rankings.append(self.store.vector_search(qvec, self.candidate_k))
        if self.mode in {"hybrid", "keyword"}:
            rankings.append(self.store.keyword_search(query, self.candidate_k))
        fused = reciprocal_rank_fusion(rankings)
        if self.reranker:
            fused = self.reranker.rerank(query, fused)
        return fused[:k]
