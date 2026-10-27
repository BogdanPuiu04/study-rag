from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from study_rag.models import Chunk, ScoredChunk

INDEX_FORMAT_VERSION = 1


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding setup than the one querying it.

    Comparing vectors from two different models gives plausible-looking garbage,
    not an error — so we fail loudly at startup instead.
    """


class VectorStore(Protocol):
    """Two retrieval signals; fusion happens in `retrieval.py`, not in the store."""

    # How the index was built: embedder, model, dim, chunking params. Set by ingest.
    meta: dict[str, Any]

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]: ...

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]: ...

    def all_chunks(self) -> list[Chunk]:
        """Every indexed chunk, ordered by (source, position). Used by the quiz generator."""
        ...

    def clear(self) -> None: ...

    def save(self) -> None: ...

    def __len__(self) -> int: ...


# Keys that must match between index and query time. Chunking params are recorded
# for reproducibility but don't make querying wrong, so they're not checked.
MUST_MATCH = ("embedder", "embed_model", "dim")


def check_compatible(index_meta: dict[str, Any], expected: dict[str, Any]) -> None:
    if not index_meta:
        raise IndexMismatchError(
            "Index has no metadata (built by an older version?). Re-run `study-rag ingest`."
        )
    diffs = {
        k: (index_meta.get(k), expected.get(k))
        for k in MUST_MATCH
        if index_meta.get(k) != expected.get(k)
    }
    if diffs:
        detail = ", ".join(f"{k}: index={a!r} vs config={b!r}" for k, (a, b) in diffs.items())
        raise IndexMismatchError(
            f"Index was built with a different embedding setup ({detail}). "
            "Re-run `study-rag ingest` or change the config back."
        )
