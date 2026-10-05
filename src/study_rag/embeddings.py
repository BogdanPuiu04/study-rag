from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised vectors, shape (len(texts), dim)."""
        ...


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


class HashEmbedder:
    """Feature-hashed bag of words + bigrams.

    Not semantic — it exists so tests and CI run in milliseconds with no model
    download. It is also a useful *baseline* in the eval report: if the real
    model doesn't beat this, something is wrong.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]:
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vec(t) for t in texts]))


class HFEmbedder:
    """Hugging Face sentence-transformers model (e.g. BGE-small, 384 dims, CPU-friendly)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32
        )


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "hf":
        return HFEmbedder(model_name)
    raise ValueError(f"unknown embedder: {kind}")
