from __future__ import annotations

from pathlib import Path

from study_rag.cache import CachedEmbedder
from study_rag.config import Settings
from study_rag.embeddings import Embedder, make_embedder
from study_rag.ingest import chunk_document, load_directory
from study_rag.llm import LLM, make_llm
from study_rag.models import Answer
from study_rag.retrieval import CrossEncoderReranker, HybridRetriever
from study_rag.store import VectorStore, make_store
from study_rag.tracing import Trace, write_trace


def download_index(s3_uri: str, dest: str) -> None:
    import boto3

    bucket, _, key = s3_uri.removeprefix("s3://").partition("/")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, dest)


class StudyRAG:
    def __init__(
        self, settings: Settings, store: VectorStore, embedder: Embedder, llm: LLM | None = None
    ) -> None:
        self.settings, self.store, self.embedder = settings, store, embedder
        self._llm = llm
        self._retrievers: dict[str, HybridRetriever] = {}

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> StudyRAG:
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
        return cls(settings, store, embedder)

    @property
    def llm(self) -> LLM:
        # Lazy: `ingest` and `eval` never need an LLM client.
        if self._llm is None:
            s = self.settings
            self._llm = make_llm(
                s.llm, model=s.claude_model, effort=s.claude_effort, region=s.aws_region
            )
        return self._llm

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
        self.store.save()
        if isinstance(self.embedder, CachedEmbedder):
            self.embedder.save()
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

    def ask(self, question: str, course: str | None = None) -> Answer:
        trace = Trace(question=question, course=course)
        try:
            with trace.timed("retrieve"):
                sources = self.retriever().retrieve(question, self.settings.top_k, course)
            trace.record_retrieved(sources)
            with trace.timed("generate"):
                text = self.llm.generate(question, sources)
            trace.model, trace.answer = self.llm.model, text
            trace.usage = getattr(self.llm, "last_usage", None)
            return Answer(text=text, sources=sources, model=self.llm.model)
        except Exception as e:
            trace.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            write_trace(trace, self.settings.trace_path)
