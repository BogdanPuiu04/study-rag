from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source file (lecture notes, slides, paper) after text extraction."""

    doc_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. `section` keeps the heading path so citations are readable."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    section: str = ""
    position: int = 0


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[ScoredChunk]
    model: str
    trace_id: str = ""
