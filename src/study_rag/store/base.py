from __future__ import annotations

from typing import Protocol

import numpy as np

from study_rag.models import Chunk, ScoredChunk


class VectorStore(Protocol):
    """Two retrieval signals; fusion happens in `retrieval.py`, not in the store."""

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]: ...

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]: ...

    def clear(self) -> None: ...

    def save(self) -> None: ...

    def __len__(self) -> int: ...
