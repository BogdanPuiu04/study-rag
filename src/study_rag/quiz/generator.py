"""Quiz generation: pick diverse chunks, ask Claude for one question per chunk with
structured output, then drop what fails the quality gates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from study_rag.llm import RefusalError, escape_tags, load_prompt
from study_rag.models import Chunk
from study_rag.pricing import cost_usd
from study_rag.quiz.models import Quiz, QuizQuestion
from study_rag.quiz.select import select_chunks
from study_rag.quiz.verifier import Review, structural_issues


class QuizError(RuntimeError):
    pass


class QuizUnavailableError(QuizError):
    """Quiz can't run in this setup: no real model (extractive mode) or no prompt written."""


class NoMaterialError(QuizError):
    """Nothing in the index is long enough to ask about (or the course filter matched none)."""


@dataclass
class QuizResult:
    questions: list[QuizQuestion]
    chunks: list[Chunk]  # the excerpts, in the order QuizQuestion.excerpt numbers them (1-based)
    rejected: list[tuple[QuizQuestion, list[str]]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] | None = None
    cost_usd: float | None = None


def build_quiz_message(chunks: list[Chunk]) -> str:
    """Numbered excerpts, one question each. Excerpts are data: a closing tag inside a
    document must not be able to end the block early (same rule as the answer prompt)."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        text = escape_tags(c.text)
        where = f"{c.source} - {c.section}" if c.section else c.source
        where = escape_tags(where).replace('"', "&quot;")
        parts.append(f'<excerpt n="{i}" source="{where}">\n{text}\n</excerpt>')
    return (
        "\n\n".join(parts)
        + f"\n\nWrite exactly one multiple-choice question for each of the {len(chunks)} "
        "excerpts above, in the same order."
    )


def generate_quiz(client, model: str, chunks: list[Chunk]) -> tuple[Quiz, dict[str, int] | None]:
    """One structured-output call: returns the parsed Quiz and the token usage."""
    system = load_prompt("quiz_system")
    if not system:
        raise QuizUnavailableError(
            "prompts/quiz_system.md is empty"
        )
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": build_quiz_message(chunks)}],
        output_format=Quiz,
    )
    if response.stop_reason == "refusal":
        raise RefusalError("the model declined to write the quiz")
    if response.stop_reason == "max_tokens":
        raise QuizError("the quiz was cut off (max_tokens): ask for fewer questions")
    usage = None
    if response.usage is not None:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    return response.parsed_output, usage


def make_quiz(
    all_chunks: list[Chunk],
    client,
    model: str,
    n: int,
    course: str | None = None,
    verify: Callable[[QuizQuestion, str], Review] | None = None,
) -> QuizResult:
    """Select chunks, generate, filter. `verify` is the optional LLM gate."""
    chunks = select_chunks(all_chunks, n, course)
    if not chunks:
        where = f" in course {course!r}" if course else ""
        raise NoMaterialError(f"no chunk long enough to quiz on{where}")
    quiz, usage = generate_quiz(client, model, chunks)

    kept: list[QuizQuestion] = []
    rejected: list[tuple[QuizQuestion, list[str]]] = []
    for q in quiz.questions:
        issues = structural_issues(q, len(chunks))
        if not issues and verify is not None:
            review = verify(q, chunks[q.excerpt - 1].text)
            if not review.ok:
                issues = review.reasons or ["rejected by verifier"]
        if issues:
            rejected.append((q, issues))
        else:
            kept.append(q)
    return QuizResult(
        questions=kept,
        chunks=chunks,
        rejected=rejected,
        model=model,
        usage=usage,
        cost_usd=cost_usd(model, usage),
    )
