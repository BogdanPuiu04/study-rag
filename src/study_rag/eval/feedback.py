"""Turn 👎 feedback into golden-set candidates.

The loop: users rate answers -> negative ratings are joined with their traces ->
you label each candidate (which source + phrase *should* have been retrieved) ->
it moves into eval/golden.jsonl. The eval grows from real failures, not guesses.

Locally the events are in .traces/queries.jsonl. On AWS they're in CloudWatch;
export them with the Logs Insights query in infra/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from study_rag.tracing import read_events


def negative_feedback_candidates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = {e["trace_id"]: e for e in events if e.get("type") == "query_trace"}
    seen: set[str] = set()
    out = []
    for fb in (e for e in events if e.get("type") == "feedback" and e.get("rating", 0) < 0):
        t = traces.get(fb["trace_id"])
        if not t or t["question"] in seen:
            continue
        seen.add(t["question"])
        out.append({
            "question": t["question"],
            "comment": fb.get("comment"),
            "answer": t.get("answer", ""),
            "retrieved": [f'{r["source"]} — {r["section"]}' for r in t.get("retrieved", [])],
            "prompt_version": t.get("prompt_version"),
            # To fill in by hand, then move the line to eval/golden.jsonl:
            "source": "TODO",
            "must_contain": "TODO",
        })
    return out


def export_candidates(traces_path: str | Path, out_path: str | Path) -> int:
    rows = negative_feedback_candidates(read_events(traces_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)
