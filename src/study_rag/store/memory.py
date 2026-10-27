from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from study_rag.embeddings import tokenize  # noqa: F401 — you'll need it for BM25
from study_rag.models import Chunk, ScoredChunk


class MemoryStore:
    """In-memory index: brute-force vector search + BM25 keyword search.

    Brute force is O(n) per query — fine up to ~100k chunks. pgvector's HNSW
    index (store/pgvector.py) is the approximate, sub-linear version of the same idea.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None  # shape (n_chunks, dim), L2-normalised rows
        self.meta: dict = {}  # how the index was built (set by the pipeline)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        # EXERCISE 3b: BM25 needs per-chunk term frequencies and corpus document
        # frequencies. Compute them here (once, at index time), not per query.

    # --- EXERCISE 3a ----------------------------------------------------------
    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]:
        """Top-k chunks by cosine similarity, highest first.

        The rows of self.vectors and query_vec are already L2-normalised.
        What does that let you replace cosine similarity with? (one numpy line)
        """
        raise NotImplementedError("exercise 3a")

    # --- EXERCISE 3b ----------------------------------------------------------
    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]:
        """Top-k chunks by BM25 score, highest first; exclude chunks scoring 0.

        Use tokenize() for both query and chunks. Start with k1=1.5, b=0.75.
        Including the chunk's `section` in its indexed text is worth trying —
        measure it with the eval.
        """
        raise NotImplementedError("exercise 3b")

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
