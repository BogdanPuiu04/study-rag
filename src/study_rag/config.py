from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime choices live here so swapping local <-> AWS is config, not code.

    Each field reads the env var `STUDY_RAG_<FIELD_NAME>`; fields whose var doesn't
    follow that pattern declare it with `Field(alias=...)`. Env is read when a Settings
    is constructed (not at import), so tests and CLIs can override it. Explicit
    keyword arguments win over the environment.
    """

    model_config = SettingsConfigDict(
        env_prefix="STUDY_RAG_",
        frozen=True,
        validate_by_name=True,  # Settings(index_path=...) keeps working next to the alias
        validate_by_alias=True,
    )

    # "hash" = deterministic, dependency-free (tests/CI); "hf" = sentence-transformers
    embedder: Literal["hash", "hf"] = "hash"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker: Literal["none", "hf"] = "none"  # "hf" = cross-encoder
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    store: Literal["memory", "pgvector"] = "memory"
    database_url: str = Field("postgresql://rag:rag@localhost:5432/rag", alias="DATABASE_URL")
    index_path: str = Field(".index/index.npz", alias="STUDY_RAG_INDEX")
    # If set (s3://bucket/key), the index is downloaded at startup, which is how Lambda gets it.
    index_s3_uri: str = Field("", alias="STUDY_RAG_INDEX_S3")
    embed_cache_path: str = Field(".index/embed_cache.npz", alias="STUDY_RAG_EMBED_CACHE")
    # AWS reindexing: where course files are uploaded, and where the embedding cache lives.
    docs_s3_uri: str = Field("", alias="STUDY_RAG_DOCS_S3")
    cache_s3_uri: str = Field("", alias="STUDY_RAG_CACHE_S3")
    # JSONL file for query traces; "-" = stdout (CloudWatch in Lambda)
    trace_path: str = Field(".traces/queries.jsonl", alias="STUDY_RAG_TRACES")
    # "extractive" ($0, offline), "claude" (Claude API), "bedrock" (Amazon Bedrock)
    llm: Literal["extractive", "claude", "bedrock"] = "extractive"
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    aws_region: str = Field("us-east-1", alias="AWS_REGION")

    chunk_size: int = Field(800, gt=0)
    chunk_overlap: int = Field(120, ge=0)
    top_k: int = Field(5, gt=0)
    candidate_k: int = Field(20, gt=0)
