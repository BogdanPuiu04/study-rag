"""End-to-end answer evaluation (costs tokens — run manually, not in CI).

Three numbers:
  faithfulness   — share of answers with zero unsupported claims (judge)
  citation_valid — share of answers whose [n] all point at a real excerpt (regex, free)
  abstention     — share of *unanswerable* questions where the system said "not covered"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from study_rag.eval.judge import judge_answer
from study_rag.llm import cited_indices
from study_rag.pipeline import StudyRAG


@dataclass
class Row:
    question: str
    answerable: bool
    answer: str
    citation_valid: bool
    faithful: bool
    abstained: bool
    unsupported_claims: list[str]


def load_questions(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


def run_answer_eval(
    rag: StudyRAG,
    answerable: list[str],
    unanswerable: list[str],
    judge_model: str,
    out_path: str | None = None,
) -> dict[str, float]:
    import anthropic

    client = anthropic.Anthropic()
    rows: list[Row] = []
    for questions, is_answerable in ((answerable, True), (unanswerable, False)):
        for q in questions:
            ans = rag.ask(q)
            verdict = judge_answer(client, judge_model, q, ans.text, ans.sources)
            valid = cited_indices(ans.text) <= set(range(1, len(ans.sources) + 1))
            rows.append(
                Row(q, is_answerable, ans.text, valid, verdict.faithful, verdict.abstained,
                    verdict.unsupported_claims)
            )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(json.dumps(asdict(r)) for r in rows) + "\n")

    ans_rows = [r for r in rows if r.answerable]
    una_rows = [r for r in rows if not r.answerable]

    def rate(rs: list[Row], attr: str) -> float:
        return sum(getattr(r, attr) for r in rs) / len(rs) if rs else 0.0

    return {
        "faithfulness": rate(ans_rows, "faithful"),
        "citation_valid": rate(ans_rows, "citation_valid"),
        "abstention_on_unanswerable": rate(una_rows, "abstained"),
        "false_abstention_on_answerable": rate(ans_rows, "abstained"),
    }
