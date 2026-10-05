"""Embedding cache: never pay (time or money) to embed the same text twice.

Keyed by sha256(model name + text), so changing the model invalidates naturally.
Re-ingesting after editing one lecture only embeds the chunks that changed.
Stored as .npz (keys + vectors), never pickle.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from study_rag.embeddings import Embedder


class CachedEmbedder:
    def __init__(self, inner: Embedder, model_name: str, path: str) -> None:
        self.inner, self.model_name, self.path = inner, model_name, Path(path)
        self.dim = inner.dim
        self._cache: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                self._cache = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))
        self.hits = self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\0{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            # One batched call for all misses — batching is what makes embedding fast.
            vectors = self.inner.embed([texts[i] for i in missing])
            for i, v in zip(missing, vectors, strict=True):
                self._cache[keys[i]] = v
        return np.stack([self._cache[k] for k in keys])

    def save(self) -> None:
        if not self._cache:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vectors=np.stack([self._cache[k] for k in keys]))
        tmp.replace(self.path)
