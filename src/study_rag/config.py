from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str):
    # Read at construction time (not import time) so tests and CLIs can override env vars.
    return field(default_factory=lambda: os.environ.get(name, default))


def _env_int(name: str, default: int):
    return field(default_factory=lambda: int(os.environ.get(name, default)))


@dataclass(frozen=True)
class Settings:
    """All runtime choices live here so swapping local <-> AWS is config, not code."""

    # "hash" = deterministic, dependency-free (tests/CI); "hf" = sentence-transformers
    embedder: str = _env("STUDY_RAG_EMBEDDER", "hash")
    embed_model: str = _env("STUDY_RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    # "none" or "hf" (cross-encoder)
    reranker: str = _env("STUDY_RAG_RERANKER", "none")
    rerank_model: str = _env("STUDY_RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # "memory" or "pgvector"
    store: str = _env("STUDY_RAG_STORE", "memory")
    database_url: str = _env("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
    index_path: str = _env("STUDY_RAG_INDEX", ".index/memory.pkl")
    # If set (s3://bucket/key), the index is downloaded at startup — how Lambda gets it.
    index_s3_uri: str = _env("STUDY_RAG_INDEX_S3", "")
    embed_cache_path: str = _env("STUDY_RAG_EMBED_CACHE", ".index/embed_cache.pkl")
    # JSONL file for query traces; "-" = stdout (CloudWatch in Lambda)
    trace_path: str = _env("STUDY_RAG_TRACES", ".traces/queries.jsonl")
    # "extractive" ($0, offline), "claude" (Claude API), "bedrock" (Amazon Bedrock)
    llm: str = _env("STUDY_RAG_LLM", "extractive")
    claude_model: str = _env("STUDY_RAG_CLAUDE_MODEL", "claude-opus-5-5")
    claude_effort: str = _env("STUDY_RAG_CLAUDE_EFFORT", "medium")
    aws_region: str = _env("AWS_REGION", "us-east-1")

    chunk_size: int = _env_int("STUDY_RAG_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("STUDY_RAG_CHUNK_OVERLAP", 120)
    top_k: int = _env_int("STUDY_RAG_TOP_K", 5)
    candidate_k: int = _env_int("STUDY_RAG_CANDIDATE_K", 20)
