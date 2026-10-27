"""Regression gate: compare an eval report against the last accepted one.

A fixed floor (`--min-recall 0.8`) only fails when quality is already bad. A baseline
fails the PR that made it *worse*, even while the number is still above the floor, and
the committed baseline file is the history of what each change did.
"""

from __future__ import annotations

import json
from pathlib import Path

Report = dict[str, dict[str, float]]


def save_baseline(report: Report, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> Report:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def regressions(
    report: Report, baseline: Report, max_drop: float = 0.03, mode: str = "hybrid"
) -> list[str]:
    """One message per metric of `mode` that fell by more than `max_drop` (absolute).

    A metric the baseline has but the report lacks counts as a regression: comparing runs
    made with different k would otherwise pass silently. An empty list means no regression.
    """
    if mode not in baseline:
        return [f"baseline has no {mode!r} results"]
    problems = []
    for metric, before in baseline[mode].items():
        after = report.get(mode, {}).get(metric)
        if after is None:
            problems.append(f"{mode} {metric}: missing from this run (baseline k differs?)")
        elif before - after > max_drop:
            problems.append(f"{mode} {metric}: {before:.3f} -> {after:.3f} (max drop {max_drop})")
    return problems
