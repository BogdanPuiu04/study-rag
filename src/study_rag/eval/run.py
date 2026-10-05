from __future__ import annotations

import json
from pathlib import Path

from study_rag.eval.metrics import GoldenItem, first_relevant_rank, mrr, recall_at_k
from study_rag.pipeline import StudyRAG


def load_golden(path: str | Path) -> list[GoldenItem]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [GoldenItem(**json.loads(line)) for line in lines if line.strip()]


def evaluate(rag: StudyRAG, golden: list[GoldenItem], k: int = 5) -> dict[str, dict[str, float]]:
    """Score each retrieval mode so the report shows what hybrid search buys us."""
    report: dict[str, dict[str, float]] = {}
    for mode in ("vector", "keyword", "hybrid"):
        retriever = rag.retriever(mode)
        ranks = [first_relevant_rank(retriever.retrieve(g.question, k), g) for g in golden]
        report[mode] = {
            "recall@1": recall_at_k(ranks, 1),
            f"recall@{k}": recall_at_k(ranks, k),
            "mrr": mrr(ranks),
        }
    return report


def format_report(report: dict[str, dict[str, float]]) -> str:
    metrics = list(next(iter(report.values())))
    header = "| mode | " + " | ".join(metrics) + " |"
    sep = "|---" * (len(metrics) + 1) + "|"
    rows = [
        f"| {mode} | " + " | ".join(f"{v:.3f}" for v in scores.values()) + " |"
        for mode, scores in report.items()
    ]
    return "\n".join([header, sep, *rows])
