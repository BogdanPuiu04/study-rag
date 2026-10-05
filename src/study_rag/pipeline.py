from __future__ import annotations

from pathlib import Path

from study_rag.config import Settings
from study_rag.embeddings import Embedder, make_embedder
from study_rag.ingest import chunk_document, load_directory
from study_rag.llm import LLM, make_llm
from study_rag.models import Answer
from study_rag.retrieval import CrossEncoderReranker, HybridRetriever
from study_rag.store import VectorStore, make_store


class StudyRAG:
    def __init__(
        self, settings: Settings, store: VectorStore, embedder: Embedder, llm: LLM | None = None
    ) -> None:
        self.settings, self.store, self.embedder = settings, store, embedder
        self._llm = llm

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> StudyRAG:
        settings = settings or Settings()
        embedder = make_embedder(settings.embedder, settings.embed_model)
        store = make_store(
            settings.store,
            dim=embedder.dim,
            index_path=settings.index_path,
            database_url=settings.database_url,
        )
        return cls(settings, store, embedder)

    @property
    def llm(self) -> LLM:
        # Lazy: `ingest` and `eval --retrieval-only` never need an LLM client.
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
        return len(chunks)

    def retriever(self, mode: str = "hybrid") -> HybridRetriever:
        reranker = (
            CrossEncoderReranker(self.settings.rerank_model)
            if self.settings.reranker == "hf"
            else None
        )
        return HybridRetriever(
            self.store, self.embedder, reranker, self.settings.candidate_k, mode=mode
        )

    def ask(self, question: str) -> Answer:
        sources = self.retriever().retrieve(question, self.settings.top_k)
        text = self.llm.generate(question, sources)
        return Answer(text=text, sources=sources, model=self.llm.model)
