"""Per-query traces and feedback events as JSON lines.

When an answer is bad, the first question is always: was the right chunk
retrieved at all? The trace answers that without re-running anything.

In Lambda (path "-" -> stdout -> CloudWatch Logs) each query trace also carries
an Embedded Metric Format block, so CloudWatch extracts latency/token/cost
metrics from the log line itself: no metrics SDK, no extra API calls.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from study_rag.models import ScoredChunk

METRICS_NAMESPACE = "StudyRAG"


@dataclass
class Trace:
    question: str
    course: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    prompt_version: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
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


def _emf(trace: Trace) -> dict[str, Any]:
    """CloudWatch Embedded Metric Format: metric values live as top-level keys."""
    values: dict[str, float] = {
        "RetrieveMs": trace.timings_ms.get("retrieve", 0.0),
        "GenerateMs": trace.timings_ms.get("generate", 0.0),
        "Errors": 1.0 if trace.error else 0.0,
    }
    if trace.usage:
        values["InputTokens"] = float(trace.usage.get("input_tokens", 0))
        values["OutputTokens"] = float(trace.usage.get("output_tokens", 0))
    if trace.cost_usd is not None:
        values["CostUSD"] = trace.cost_usd
    units = {"RetrieveMs": "Milliseconds", "GenerateMs": "Milliseconds"}
    return {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": METRICS_NAMESPACE,
                "Dimensions": [["Model"]],
                "Metrics": [{"Name": k, "Unit": units.get(k, "None")} for k in values],
            }],
        },
        "Model": trace.model or "none",
        **values,
    }


def _write(record: dict[str, Any], path: str | None) -> None:
    line = json.dumps(record, default=str)
    if not path or path == "-":
        print(line, file=sys.stdout, flush=True)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_trace(trace: Trace, path: str | None) -> None:
    """path=None or "-" -> stdout with EMF metrics (CloudWatch); otherwise append to JSONL."""
    record: dict[str, Any] = {"type": "query_trace", "ts": time.time(), **asdict(trace)}
    if not path or path == "-":
        record = {**_emf(trace), **record}
    _write(record, path)


def write_feedback(
    trace_id: str, rating: int, comment: str | None, path: str | None
) -> None:
    _write(
        {"type": "feedback", "ts": time.time(), "trace_id": trace_id,
         "rating": rating, "comment": comment},
        path,
    )


def read_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
