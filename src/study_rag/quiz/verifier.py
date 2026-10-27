"""Quality gate for generated questions: a cheap structural check (implemented) and an
LLM check that the excerpt really supports the keyed answer."""

from __future__ import annotations

from pydantic import BaseModel

from study_rag.llm import RefusalError, escape_tags, load_prompt
from study_rag.quiz.models import QuizQuestion


def structural_issues(q: QuizQuestion, n_excerpts: int) -> list[str]:
    """Everything wrong with `q` that can be seen without reading the course material.

    Free, instant and catches the model's format slips, which is why it runs before the
    expensive check. An empty list means the question is well-formed (not that it is good).
    """
    issues: list[str] = []
    if not q.question.strip():
        issues.append("empty question")
    if len(q.options) != 4:
        issues.append(f"expected 4 options, got {len(q.options)}")
    if any(not o.strip() for o in q.options):
        issues.append("empty option")
    if len({o.strip().lower() for o in q.options}) != len(q.options):
        issues.append("duplicate options")
    if not 0 <= q.correct_index < len(q.options):
        issues.append(f"correct_index {q.correct_index} out of range")
    if not q.explanation.strip():
        issues.append("empty explanation")
    if not 1 <= q.excerpt <= n_excerpts:
        issues.append(f"excerpt {q.excerpt} does not exist (1..{n_excerpts})")
    return issues


class Review(BaseModel):
    """What the verifier returns (also the structured-output schema)."""

    ok: bool  # True iff the question should be kept
    reasons: list[str]  # why not, when ok is False; empty otherwise


def verify_question(client, model: str, question: QuizQuestion, excerpt_text: str) -> Review:
    """Ask Claude whether `question` is a fair question about `excerpt_text`.

    Rejects a question whose keyed answer the excerpt doesn't support, where another
    option is also defensible, or that can be answered without the excerpt. Runs after the
    free structural check, so only well-formed questions cost a call.
    """
    options = "\n".join(f"{'ABCD'[i]}) {escape_tags(o)}" for i, o in enumerate(question.options))
    keyed = "ABCD"[question.correct_index] if 0 <= question.correct_index < 4 else "?"
    message = (
        f"<excerpt>\n{escape_tags(excerpt_text)}\n</excerpt>\n\n"
        f"Question: {escape_tags(question.question)}\n{options}\n"
        f"Keyed answer: {keyed}\n"
        f"Explanation: {escape_tags(question.explanation)}\n\n"
        "Review this question."
    )
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=load_prompt("quiz_verifier_system"),
        messages=[{"role": "user", "content": message}],
        output_format=Review,
    )
    if response.stop_reason == "refusal":
        raise RefusalError("the verifier declined to review this question")
    if response.stop_reason == "max_tokens":
        # An unfinished review is not an approval.
        return Review(ok=False, reasons=["verifier output was cut off"])
    review: Review = response.parsed_output
    return review if not review.ok else review.model_copy(update={"reasons": []})
