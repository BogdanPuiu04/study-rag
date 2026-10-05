"""Per-query traces as JSON lines.

When an answer is bad, the first question is always: was the right chunk
retrieved at all? The trace answers that without re-running anything.
In Lambda, stdout goes to CloudWatch Logs, where the same JSON is queryable
with Logs Insights.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from study_rag.models import ScoredChunk


@dataclass
class Trace:
    question: str
    course: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] | None = None
    answer: str = ""
    error: str | None = None

    @contextmanager
    def timed(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[stage] = round((time.perf_counter() - start) * 1000, 1)

    def record_retrieved(self, chunks: list[ScoredChunk]) -> None:
        self.retrieved = [
            {
                "rank": i,
                "chunk_id": sc.chunk.chunk_id,
                "source": sc.chunk.source,
                "section": sc.chunk.section,
                "score": round(sc.score, 5),
            }
            for i, sc in enumerate(chunks, start=1)
        ]


def write_trace(trace: Trace, path: str | None) -> None:
    """path=None or "-" -> stdout (CloudWatch in Lambda); otherwise append to a JSONL file."""
    line = json.dumps({"type": "query_trace", "ts": time.time(), **trace.__dict__})
    if not path or path == "-":
        print(line, file=sys.stdout, flush=True)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
