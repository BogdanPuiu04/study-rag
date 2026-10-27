"""LLM-as-judge for answer faithfulness.

Retrieval metrics tell you whether the right excerpt was *found*. They say
nothing about whether the generated answer *sticks to it*. The judge reads
(question, excerpts, answer) and checks every claim against the excerpts.
"""

from __future__ import annotations

from pydantic import BaseModel

from study_rag.llm import RefusalError, escape_tags, format_excerpts, load_prompt
from study_rag.models import ScoredChunk


class Verdict(BaseModel):
    """What the judge returns (also the structured-output schema). Extending it means
    updating answer_eval.py too."""

    unsupported_claims: list[str]  # claims in the answer NOT backed by any excerpt
    abstained: bool  # the answer says the materials don't cover the question
    faithful: bool  # True iff no unsupported claims


def judge_answer(
    client, model: str, question: str, answer: str, chunks: list[ScoredChunk]
) -> Verdict:
    """Ask Claude to grade one answer against the excerpts it was given.

    Structured output (`messages.parse` with the Verdict schema) means the verdict is
    validated, never parsed out of free text. The judge sees exactly the excerpts the
    answering model saw, so "unsupported" means unsupported by *those*, not by the course.
    """
    message = (
        f"{format_excerpts(chunks)}\n\n"
        f"<question>\n{escape_tags(question)}\n</question>\n\n"
        f"<answer>\n{escape_tags(answer)}\n</answer>\n\n"
        "Grade the answer against the excerpts."
    )
    response = client.messages.parse(
        model=model,
        max_tokens=8000,
        system=load_prompt("judge_system"),
        messages=[{"role": "user", "content": message}],
        output_format=Verdict,
    )
    if response.stop_reason == "refusal":
        raise RefusalError("the judge declined to grade this answer")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the judge's verdict was cut off (max_tokens)")
    verdict: Verdict = response.parsed_output
    # `faithful` is defined by the claims list; don't let the two disagree.
    return verdict.model_copy(update={"faithful": not verdict.unsupported_claims})
