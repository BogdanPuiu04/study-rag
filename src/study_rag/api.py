"""HTTP API. Same app runs locally (uvicorn) and in Lambda (via Lambda Web Adapter,
so there is no Lambda-specific code here at all — S3 events arrive as POST /events)."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from study_rag import __version__
from study_rag.config import Settings
from study_rag.llm import RefusalError
from study_rag.models import ScoredChunk
from study_rag.pipeline import StudyRAG
from study_rag.quiz import NoMaterialError, QuizError, QuizUnavailableError
from study_rag.s3sync import IndexWatcher, reindex
from study_rag.store.base import IndexMismatchError

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="study-rag", version=__version__)

_lock = threading.Lock()
_rag: StudyRAG | None = None
_watcher: IndexWatcher | None = None


def get_rag() -> StudyRAG:
    """One StudyRAG per process (= per Lambda instance), reloaded when the index in S3 changes."""
    global _rag, _watcher
    with _lock:
        if _watcher is None:
            _watcher = IndexWatcher(Settings().index_s3_uri)
        if _rag is None or _watcher.changed():
            _rag = StudyRAG.from_settings(Settings())
            _watcher.mark_loaded()
        return _rag


# --- schemas ----------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    course: str | None = None


class Source(BaseModel):
    n: int
    source: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    trace_id: str


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=8, max_length=64)
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=2000)


class QuizRequest(BaseModel):
    n: int = Field(default=5, ge=1, le=10)
    course: str | None = None


class QuizItem(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source: str
    section: str


class QuizResponse(BaseModel):
    questions: list[QuizItem]
    rejected: int
    model: str
    cost_usd: float | None


def _sources(chunks: list[ScoredChunk]) -> list[Source]:
    return [
        Source(n=i, source=sc.chunk.source, section=sc.chunk.section, score=sc.score)
        for i, sc in enumerate(chunks, start=1)
    ]


# --- routes -----------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, object]:
    try:
        rag = get_rag()
    except IndexMismatchError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {
        "status": "ok",
        "version": __version__,
        "chunks": len(rag.store),
        "index": rag.store.meta,
        "prompt_version": rag.prompt_version,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        ans = get_rag().ask(req.question, course=req.course)
    except RefusalError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return AskResponse(
        answer=ans.text, model=ans.model, trace_id=ans.trace_id, sources=_sources(ans.sources)
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events: `sources` first, then `delta`s, then `done` (or `error`)."""
    trace_id, sources, pieces = get_rag().ask_stream(req.question, course=req.course)

    def sse(event: str, data: object) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def events() -> Iterator[str]:
        yield sse("sources", [s.model_dump() for s in _sources(sources)])
        try:
            for piece in pieces:
                yield sse("delta", {"text": piece})
        except Exception as e:  # headers are already sent: report in-band
            yield sse("error", {"detail": f"{type(e).__name__}: {e}"})
            return
        yield sse("done", {"trace_id": trace_id})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/quiz", response_model=QuizResponse)
def quiz(req: QuizRequest) -> QuizResponse:
    try:
        result = get_rag().quiz(req.n, course=req.course)
    except QuizUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except NoMaterialError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RefusalError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except QuizError as e:  # truncated output etc.: the upstream model misbehaved
        raise HTTPException(status_code=502, detail=str(e)) from e
    items = []
    for q in result.questions:
        chunk = result.chunks[q.excerpt - 1]
        items.append(
            QuizItem(
                **q.model_dump(include={"question", "options", "correct_index", "explanation"}),
                source=chunk.source,
                section=chunk.section,
            )
        )
    return QuizResponse(
        questions=items,
        rejected=len(result.rejected),
        model=result.model,
        cost_usd=result.cost_usd,
    )


@app.post("/feedback", status_code=204)
def feedback(req: FeedbackRequest) -> None:
    get_rag().record_feedback(req.trace_id, req.rating, req.comment)


@app.post("/events")
def events(payload: dict) -> dict[str, object]:
    """Non-HTTP Lambda events (Lambda Web Adapter pass-through). S3 uploads -> reindex."""
    records = payload.get("Records") or []
    keys = [r["s3"]["object"]["key"] for r in records if r.get("eventSource") == "aws:s3"]
    if not keys:
        return {"ignored": True}
    result = reindex(Settings())
    global _rag
    with _lock:
        _rag = None  # reload on next request
    return {"reindexed": result, "trigger": keys}
