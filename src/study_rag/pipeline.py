from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from study_rag.cache import CachedEmbedder
from study_rag.config import Settings
from study_rag.embeddings import Embedder, make_embedder
from study_rag.ingest import chunk_document, load_directory
from study_rag.llm import LLM, make_llm, prompt_version
from study_rag.models import Answer, ScoredChunk
from study_rag.pricing import cost_usd
from study_rag.retrieval import CrossEncoderReranker, HybridRetriever
from study_rag.store import VectorStore, make_store
from study_rag.store.base import INDEX_FORMAT_VERSION, check_compatible
from study_rag.tracing import Trace, write_feedback, write_trace


def _split_s3(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def download_index(s3_uri: str, dest: str) -> bool:
    """Fetch the index; False if it doesn't exist yet (fresh deploy, no docs uploaded)."""
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = _split_s3(s3_uri)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    try:
        boto3.client("s3").download_file(bucket, key, dest)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return False
        raise


def index_meta(settings: Settings, embedder: Embedder) -> dict[str, Any]:
    """What the index records about how it was built (see store/base.check_compatible)."""
    return {
        "format_version": INDEX_FORMAT_VERSION,
        "embedder": settings.embedder,
        "embed_model": settings.embed_model if settings.embedder != "hash" else "hash",
        "dim": embedder.dim,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


class StudyRAG:
    def __init__(
        self, settings: Settings, store: VectorStore, embedder: Embedder, llm: LLM | None = None
    ) -> None:
        self.settings, self.store, self.embedder = settings, store, embedder
        self._llm = llm
        self._retrievers: dict[str, HybridRetriever] = {}
        self.prompt_version = prompt_version("answer_system")

    @classmethod
    def from_settings(cls, settings: Settings | None = None, check_index: bool = True) -> StudyRAG:
        settings = settings or Settings()
        if settings.index_s3_uri:
            download_index(settings.index_s3_uri, settings.index_path)
        embedder: Embedder = make_embedder(settings.embedder, settings.embed_model)
        if settings.embedder != "hash":  # hashing is cheaper than a cache lookup
            embedder = CachedEmbedder(embedder, settings.embed_model, settings.embed_cache_path)
        store = make_store(
            settings.store,
            dim=embedder.dim,
            index_path=settings.index_path,
            database_url=settings.database_url,
        )
        rag = cls(settings, store, embedder)
        # An empty store is fine (nothing ingested yet); a non-empty one must match.
        if check_index and len(store):
            check_compatible(store.meta, index_meta(settings, embedder))
        return rag

    @property
    def llm(self) -> LLM:
        # Lazy: `ingest` and `eval` never need an LLM client.
        if self._llm is None:
            s = self.settings
            self._llm = make_llm(
                s.llm, model=s.claude_model, effort=s.claude_effort, region=s.aws_region
            )
        return self._llm

    # --- indexing -----------------------------------------------------------------
    def ingest(self, root: str | Path, reset: bool = True) -> int:
        if reset:
            self.store.clear()
        docs = load_directory(root)
        chunks = [
            c
            for d in docs
            for c in chunk_document(d, self.settings.chunk_size, self.settings.chunk_overlap)
        ]
        if chunks:
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        self.store.meta = index_meta(self.settings, self.embedder)
        self.store.save()
        if isinstance(self.embedder, CachedEmbedder):
            self.embedder.save()
        self._retrievers.clear()
        return len(chunks)

    def retriever(self, mode: str = "hybrid") -> HybridRetriever:
        # Cached: loading a cross-encoder takes seconds; do it once per process.
        if mode not in self._retrievers:
            reranker = (
                CrossEncoderReranker(self.settings.rerank_model)
                if self.settings.reranker == "hf"
                else None
            )
            self._retrievers[mode] = HybridRetriever(
                self.store, self.embedder, reranker, self.settings.candidate_k, mode=mode
            )
        return self._retrievers[mode]

    # --- querying -----------------------------------------------------------------
    def _new_trace(self, question: str, course: str | None) -> Trace:
        return Trace(question=question, course=course, prompt_version=self.prompt_version)

    def _retrieve(self, trace: Trace, question: str, course: str | None) -> list[ScoredChunk]:
        with trace.timed("retrieve"):
            sources = self.retriever().retrieve(question, self.settings.top_k, course)
        trace.record_retrieved(sources)
        return sources

    def _finish(self, trace: Trace, text: str) -> None:
        trace.model, trace.answer = self.llm.model, text
        trace.usage = getattr(self.llm, "last_usage", None)
        trace.cost_usd = cost_usd(self.llm.model, trace.usage)

    def ask(self, question: str, course: str | None = None) -> Answer:
        trace = self._new_trace(question, course)
        try:
            sources = self._retrieve(trace, question, course)
            with trace.timed("generate"):
                text = self.llm.generate(question, sources)
            self._finish(trace, text)
            return Answer(
                text=text, sources=sources, model=self.llm.model, trace_id=trace.trace_id
            )
        except Exception as e:
            trace.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            write_trace(trace, self.settings.trace_path)

    def ask_stream(
        self, question: str, course: str | None = None
    ) -> tuple[str, list[ScoredChunk], Iterator[str]]:
        """Retrieve now, generate lazily. Returns (trace_id, sources, text pieces).
        The trace is written when the stream is exhausted (or fails)."""
        trace = self._new_trace(question, course)
        try:
            sources = self._retrieve(trace, question, course)
        except Exception as e:
            trace.error = f"{type(e).__name__}: {e}"
            write_trace(trace, self.settings.trace_path)
            raise

        def pieces() -> Iterator[str]:
            parts: list[str] = []
            try:
                with trace.timed("generate"):
                    for piece in self.llm.generate_stream(question, sources):
                        parts.append(piece)
                        yield piece
                self._finish(trace, "".join(parts))
            except Exception as e:
                trace.error = f"{type(e).__name__}: {e}"
                raise
            finally:
                write_trace(trace, self.settings.trace_path)

        return trace.trace_id, sources, pieces()

    def record_feedback(self, trace_id: str, rating: int, comment: str | None = None) -> None:
        write_feedback(trace_id, rating, comment, self.settings.trace_path)
