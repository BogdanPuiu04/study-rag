"""HTTP API. Same app runs locally (uvicorn) and in Lambda (via Lambda Web Adapter,
so there is no Lambda-specific code here at all)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from study_rag import __version__
from study_rag.config import Settings
from study_rag.llm import RefusalError
from study_rag.pipeline import StudyRAG

app = FastAPI(title="study-rag", version=__version__)


@lru_cache(maxsize=1)
def get_rag() -> StudyRAG:
    # Loaded once per process (= once per Lambda cold start), not per request.
    return StudyRAG.from_settings(Settings())


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


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "version": __version__, "chunks": len(get_rag().store)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        ans = get_rag().ask(req.question, course=req.course)
    except RefusalError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return AskResponse(
        answer=ans.text,
        model=ans.model,
        sources=[
            Source(n=i, source=sc.chunk.source, section=sc.chunk.section, score=sc.score)
            for i, sc in enumerate(ans.sources, start=1)
        ],
    )
