"""Quality gate for generated questions: a cheap structural check (implemented) and an
LLM check that the excerpt really supports the keyed answer (EXERCISE 12b)."""

from __future__ import annotations

from pydantic import BaseModel

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
    """What the verifier returns. Extend it if your checks need more fields."""

    ok: bool  # True iff the question should be kept
    reasons: list[str]  # why not, when ok is False; empty otherwise


def verify_question(client, model: str, question: QuizQuestion, excerpt_text: str) -> Review:
    """EXERCISE 12b. Ask Claude whether `question` is a good question about `excerpt_text`.

    See docs/EXERCISES.md §12. Reject a question when the keyed answer is not supported by
    the excerpt, when more than one option is defensible, or when it can be answered without
    reading the excerpt at all.

    - system prompt: load_prompt("quiz_verifier_system") from study_rag.llm
    - structured output: client.messages.parse(..., output_format=Review)
    - check stop_reason before trusting parsed_output (refusal -> RefusalError)
    """
    raise NotImplementedError("exercise 12b")
