"""Vector search, BM25 and Reciprocal Rank Fusion."""

import numpy as np

from study_rag.embeddings import HashEmbedder
from study_rag.models import Chunk, ScoredChunk
from study_rag.retrieval import reciprocal_rank_fusion
from study_rag.store.memory import MemoryStore

TEXTS = [
    "Backpropagation applies the chain rule layer by layer.",
    "Dropout randomly zeroes activations during training.",
    "The CAP theorem: consistency or availability under partition.",
    "Dropout dropout dropout regularisation.",
]


def make_store() -> MemoryStore:
    chunks = [Chunk(f"c{i}", "d", t, "s.md") for i, t in enumerate(TEXTS)]
    store = MemoryStore()
    store.add(chunks, HashEmbedder().embed(TEXTS))
    return store


# vector search
def test_vector_search_exact_match_ranks_first_and_sorted():
    store = make_store()
    q = HashEmbedder().embed([TEXTS[2]])[0]
    res = store.vector_search(q, k=3)
    assert len(res) == 3
    assert res[0].chunk.chunk_id == "c2"
    assert res[0].score == np.float32(1.0) or abs(res[0].score - 1.0) < 1e-5
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)


# BM25
def test_keyword_search_finds_term_and_excludes_zero_scores():
    res = make_store().keyword_search("chain rule", k=10)
    assert [r.chunk.chunk_id for r in res] == ["c0"]


def test_keyword_search_term_frequency_saturates_not_explodes():
    # BM25's k1 caps how much repeating a word helps; the longer, repetitive
    # chunk c3 should not beat c1 by a huge factor.
    res = make_store().keyword_search("dropout", k=10)
    assert {r.chunk.chunk_id for r in res} == {"c1", "c3"}
    assert res[0].score / res[1].score < 2.0


def test_keyword_search_unknown_term_returns_empty():
    assert make_store().keyword_search("kubernetes", k=5) == []


# RRF
def s(cid: str) -> ScoredChunk:
    return ScoredChunk(Chunk(cid, "d", cid, "s"), 0.0)


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([[s("a"), s("b")], [s("b"), s("c")]])
    assert [f.chunk.chunk_id for f in fused] == ["b", "a", "c"]
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9


def test_rrf_deduplicates():
    fused = reciprocal_rank_fusion([[s("a")], [s("a")], [s("a")]])
    assert len(fused) == 1
