from __future__ import annotations

from study_rag.store.base import VectorStore
from study_rag.store.memory import MemoryStore


def make_store(kind: str, *, dim: int, index_path: str, database_url: str) -> VectorStore:
    if kind == "memory":
        return MemoryStore.load(index_path) or MemoryStore(path=index_path)
    if kind == "pgvector":
        from study_rag.store.pgvector import PgVectorStore

        return PgVectorStore(database_url, dim=dim)
    raise ValueError(f"unknown store: {kind}")


__all__ = ["VectorStore", "MemoryStore", "make_store"]
