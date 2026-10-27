"""Study assistant over your course materials."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from study_rag.config import Settings
from study_rag.eval.baseline import load_baseline, regressions, save_baseline
from study_rag.eval.run import evaluate, format_report, load_golden
from study_rag.pipeline import StudyRAG

# show_locals=False: tracebacks would otherwise print Settings, API keys included.
app = typer.Typer(
    help=__doc__, no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False
)


def _rag() -> StudyRAG:
    return StudyRAG.from_settings(Settings())


@app.command()
def ingest(path: str) -> None:
    """Index a folder of notes / slides / PDFs."""
    n = _rag().ingest(path)
    typer.echo(f"Indexed {n} chunks from {path}")


@app.command()
def ask(
    question: str,
    course: Annotated[
        str | None, typer.Option(help="Restrict to one course folder, e.g. machine-learning.")
    ] = None,
) -> None:
    """Ask a question about your course materials."""
    answer = _rag().ask(question, course=course)
    typer.echo(f"{answer.text}\n\n(trace {answer.trace_id})\nSources:")
    for i, sc in enumerate(answer.sources, start=1):
        section = f" - {sc.chunk.section}" if sc.chunk.section else ""
        typer.echo(f"  [{i}] {sc.chunk.source}{section}  (score {sc.score:.3f})")


@app.command()
def quiz(
    n: Annotated[int, typer.Option(help="Number of questions.")] = 5,
    course: Annotated[str | None, typer.Option(help="Restrict to one course folder.")] = None,
    verify: Annotated[bool, typer.Option(help="Also run the LLM verifier (exercise 12b).")] = False,
) -> None:
    """Multiple-choice quiz from your course materials, with citations (costs tokens)."""
    from study_rag.quiz import QuizError

    try:
        result = _rag().quiz(n, course=course, verify=verify)
    except QuizError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    for i, q in enumerate(result.questions, start=1):
        chunk = result.chunks[q.excerpt - 1]
        typer.echo(f"\n{i}. {q.question}")
        for letter, option in zip("ABCD", q.options, strict=False):
            typer.echo(f"   {letter}) {option}")
        typer.echo(f"   Answer: {'ABCD'[q.correct_index]}. {q.explanation}")
        typer.echo(f"   Source: {chunk.source}" + (f" - {chunk.section}" if chunk.section else ""))
    cost = f", ${result.cost_usd:.4f}" if result.cost_usd is not None else ""
    typer.echo(f"\n{len(result.questions)} questions, {len(result.rejected)} rejected{cost}")


@app.command("eval")
def eval_retrieval(
    golden: str,
    k: Annotated[int, typer.Option("-k", help="Cutoff for recall@k / MRR.")] = 5,
    min_recall: Annotated[float, typer.Option(help="Fail below this (CI gate).")] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the raw report.")] = False,
    baseline: Annotated[
        str | None, typer.Option(help="Fail if hybrid metrics dropped vs this saved report.")
    ] = None,
    max_drop: Annotated[float, typer.Option(help="Allowed absolute drop per metric.")] = 0.03,
    save: Annotated[
        str | None, typer.Option("--save-baseline", help="Write this run's report here.")
    ] = None,
) -> None:
    """Retrieval metrics against a golden set (free)."""
    report = evaluate(_rag(), load_golden(golden), k)
    typer.echo(json.dumps(report, indent=2) if as_json else format_report(report))
    hybrid_recall = report["hybrid"][f"recall@{k}"]
    if save:
        save_baseline(report, save)
        typer.echo(f"Saved baseline to {save}", err=True)
    failed = False
    if hybrid_recall < min_recall:
        typer.echo(f"FAIL: hybrid recall@{k}={hybrid_recall:.3f} < {min_recall}", err=True)
        failed = True
    if baseline:
        for problem in regressions(report, load_baseline(baseline), max_drop):
            typer.echo(f"FAIL: regression vs {baseline}: {problem}", err=True)
            failed = True
    if failed:
        raise typer.Exit(1)


@app.command("eval-answers")
def eval_answers(
    golden: str,
    unanswerable: str = "eval/unanswerable.jsonl",
    limit: Annotated[int, typer.Option(help="Cap questions to cap cost.")] = 10,
    judge_model: str = "claude-opus-5-5",
    out: str = ".traces/answer_eval.jsonl",
) -> None:
    """LLM-as-judge answer eval (costs tokens)."""
    # Lazy: only this command needs the Anthropic SDK.
    from study_rag.eval.answer_eval import load_questions, run_answer_eval

    rag = _rag()
    if rag.settings.llm == "extractive":
        typer.echo(
            "Set STUDY_RAG_LLM=claude or bedrock - judging extractive output is pointless.",
            err=True,
        )
        raise typer.Exit(2)
    answerable = [g.question for g in load_golden(golden)][:limit]
    unanswerable_qs = load_questions(unanswerable)[: max(1, limit // 2)]
    typer.echo(
        f"Judging {len(answerable) + len(unanswerable_qs)} answers with {judge_model}...",
        err=True,
    )
    report = run_answer_eval(rag, answerable, unanswerable_qs, judge_model, out)
    typer.echo(json.dumps(report, indent=2))


@app.command("feedback-candidates")
def feedback_candidates(
    traces: Annotated[str | None, typer.Option(help="Defaults to STUDY_RAG_TRACES.")] = None,
    out: str = "eval/candidates.jsonl",
) -> None:
    """Thumbs-down-rated questions -> golden-set candidates."""
    from study_rag.eval.feedback import export_candidates

    n = export_candidates(traces or Settings().trace_path, out)
    typer.echo(
        f"Wrote {n} candidates to {out}: label `source` and `must_contain`, "
        "then move them into eval/golden.jsonl"
    )


if __name__ == "__main__":
    app()
