from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from study_rag.models import Chunk, ScoredChunk

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  text PRIMARY KEY,
    doc_id    text NOT NULL,
    source    text NOT NULL,
    section   text NOT NULL DEFAULT '',
    position  int  NOT NULL,
    text      text NOT NULL,
    embedding vector({dim}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS
              (to_tsvector('english', section || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""

_COLS = "chunk_id, doc_id, source, section, position, text"


def _row_to_scored(row: tuple) -> ScoredChunk:
    chunk_id, doc_id, source, section, position, text, score = row
    return ScoredChunk(Chunk(chunk_id, doc_id, text, source, section, position), float(score))


class PgVectorStore:
    """Postgres does both halves: HNSW for vectors, GIN full-text for keywords."""

    def __init__(self, dsn: str, dim: int) -> None:
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(_SCHEMA.format(dim=dim))

    def __len__(self) -> int:
        return self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO chunks ({_COLS}, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, "
                "embedding = EXCLUDED.embedding, section = EXCLUDED.section",
                [
                    (c.chunk_id, c.doc_id, c.source, c.section, c.position, c.text, v)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]:
        rows = self.conn.execute(
            f"SELECT {_COLS}, 1 - (embedding <=> %s) AS score FROM chunks "
            "ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, query_vec, k),
        ).fetchall()
        return [_row_to_scored(r) for r in rows]

    def keyword_search(self, query: str, k: int) -> list[ScoredChunk]:
        rows = self.conn.execute(
            f"SELECT {_COLS}, ts_rank_cd(tsv, q) AS score "
            "FROM chunks, websearch_to_tsquery('english', %s) q "
            "WHERE tsv @@ q ORDER BY score DESC LIMIT %s",
            (query, k),
        ).fetchall()
        return [_row_to_scored(r) for r in rows]

    def clear(self) -> None:
        self.conn.execute("TRUNCATE chunks")

    def save(self) -> None:
        pass  # autocommit
