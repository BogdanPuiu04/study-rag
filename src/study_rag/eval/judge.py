"""EXERCISE 7 — LLM-as-judge for answer faithfulness.

Retrieval metrics tell you whether the right excerpt was *found*. They say
nothing about whether the generated answer *sticks to it*. The judge reads
(question, excerpts, answer) and checks every claim against the excerpts.
"""

from __future__ import annotations

from pydantic import BaseModel

from study_rag.models import ScoredChunk


class Verdict(BaseModel):
    """What the judge returns. Feel free to extend — but update answer_eval.py too."""

    unsupported_claims: list[str]  # claims in the answer NOT backed by any excerpt
    abstained: bool  # the answer says the materials don't cover the question
    faithful: bool  # True iff no unsupported claims


def judge_answer(
    client, model: str, question: str, answer: str, chunks: list[ScoredChunk]
) -> Verdict:
    """Ask Claude to grade one answer and return a validated Verdict.

    - system prompt: load_prompt("judge_system") from study_rag.llm
    - reuse build_user_message() for the excerpts, then append the answer under test
    - use structured outputs so you never parse free text:
        client.messages.parse(..., output_format=Verdict) -> response.parsed_output
    - check stop_reason before trusting parsed_output (refusal / max_tokens)
    """
    raise NotImplementedError("exercise 7")
