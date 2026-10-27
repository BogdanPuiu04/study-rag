from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from study_rag.embeddings import tokenize
from study_rag.models import Chunk, ScoredChunk

# BM25 parameters. k1 controls how fast repeating a term stops helping (term-frequency
# saturation); b controls how much long chunks are penalised (length normalisation).
# 1.5 / 0.75 are the usual defaults; tune only with the eval.
BM25_K1 = 1.5
BM25_B = 0.75


class MemoryStore:
    """In-memory index: brute-force vector search + BM25 keyword search.

    Brute force is O(n) per query: fine up to ~100k chunks, and exact rather than
    approximate. pgvector's HNSW index (store/pgvector.py) is the approximate,
    sub-linear version of the same idea.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None  # shape (n_chunks, dim), L2-normalised rows
        self.meta: dict = {}  # how the index was built (set by the pipeline)
        # BM25 statistics, computed once at index time so a query only does lookups.
        self._term_freqs: list[Counter[str]] = []  # per chunk: term -> count
        self._doc_lens: list[int] = []  # per chunk: number of tokens
        self._doc_freq: Counter[str] = Counter()  # term -> number of chunks containing it

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        for chunk in chunks:
            tf = Counter(tokenize(chunk.text))
            self._term_freqs.append(tf)
            self._doc_lens.append(sum(tf.values()))
            self._doc_freq.update(tf.keys())

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]:
        """Top-k chunks by cosine similarity, highest first.

        Rows and query are L2-normalised, so cosine similarity is a plain dot product and
        one matrix-vector product scores every chunk at once.
        """
        if self.vectors is None or k <= 0:
            return []
        scores = self.vectors @ query_vec
        return self._top_k(scores, k)

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]:
        """Top-k chunks by BM25 score, highest first; chunks scoring 0 are excluded.

        Only the chunk text is indexed. Adding the `section` heading path is a candidate
        improvement to measure with the eval, not a default.
        """
        n = len(self.chunks)
        terms = set(tokenize(query))
        if not n or not terms or k <= 0:
            return []
        avg_len = sum(self._doc_lens) / n or 1.0
        scores = np.zeros(n)
        for term in terms:
            df = self._doc_freq.get(term, 0)
            if not df:
                continue
            # The "+1" form of IDF stays positive even for terms in most chunks.
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf_counts in enumerate(self._term_freqs):
                tf = tf_counts.get(term, 0)
                if tf:
                    norm = BM25_K1 * (1 - BM25_B + BM25_B * self._doc_lens[i] / avg_len)
                    scores[i] += idf * tf * (BM25_K1 + 1) / (tf + norm)
        return [sc for sc in self._top_k(scores, k) if sc.score > 0]

    def _top_k(self, scores: np.ndarray, k: int) -> list[ScoredChunk]:
        k = min(k, len(scores))
        # argpartition finds the top k in O(n); only those k are then sorted.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in top]

    # --- plumbing -------------------------------------------------------------
    # Persistence format: one .npz with the vectors plus JSON strings for chunks and
    # metadata. No pickle — loading a pickle executes code, so a tampered index file
    # in S3 would be remote code execution. Keyword-index structures aren't saved:
    # load() rebuilds them by replaying add(), which keeps the format independent
    # of how BM25 is implemented.

    def all_chunks(self) -> list[Chunk]:
        return sorted(self.chunks, key=lambda c: (c.source, c.position))

    def clear(self) -> None:
        self.__init__(self.path)  # type: ignore[misc]  # deliberate: resets every field

    def save(self) -> None:
        if not self.path:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        dim = self.meta.get("dim", 0)
        vectors = self.vectors if self.vectors is not None else np.zeros((0, dim), np.float32)
        tmp = Path(f"{self.path}.tmp.npz")
        np.savez_compressed(
            tmp,
            vectors=vectors.astype(np.float32),
            chunks=np.array(json.dumps([asdict(c) for c in self.chunks])),
            meta=np.array(json.dumps(self.meta)),
        )
        tmp.replace(self.path)  # atomic: a crash mid-write never leaves a half index

    @classmethod
    def load(cls, path: str) -> MemoryStore | None:
        if not Path(path).exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            chunks = [Chunk(**c) for c in json.loads(str(data["chunks"]))]
            meta = json.loads(str(data["meta"]))
            vectors = data["vectors"]
        store = cls(path)
        store.meta = meta
        if chunks:
            store.add(chunks, vectors)
        return store
