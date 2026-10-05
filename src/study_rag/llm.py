"""Answer generation. Three backends behind one interface:

- extractive: $0, offline — returns the best-matching passages verbatim. Used in CI.
- claude:     Claude API (ANTHROPIC_API_KEY or `ant auth login`).
- bedrock:    Claude on Amazon Bedrock (AWS credentials; IAM role in AWS).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from study_rag.models import ScoredChunk

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def cited_indices(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


# --- EXERCISE 5b --------------------------------------------------------------


def build_user_message(question: str, chunks: list[ScoredChunk]) -> str:
    """Build the user turn: the numbered excerpts (1-based), then the question.

    Requirements (tests/test_pipeline.py):
      - each excerpt clearly delimited and numbered so [n] citations map back
      - include source and section so the model (and you, when debugging) see provenance
      - the question comes AFTER the excerpts. Why does order matter for long context?
    """
    raise NotImplementedError("exercise 5b")


class LLM(Protocol):
    model: str

    def generate(self, question: str, chunks: list[ScoredChunk]) -> str: ...


class ExtractiveLLM:
    """No model call: stitches the top passages together with citations."""

    model = "extractive"

    def generate(self, question: str, chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return "I couldn't find anything about that in your course materials."
        return "\n\n".join(f"{sc.chunk.text.strip()} [{i}]" for i, sc in enumerate(chunks[:2], 1))


class RefusalError(RuntimeError):
    pass


# --- EXERCISE 5c --------------------------------------------------------------


class ClaudeLLM:
    def __init__(self, model: str, effort: str = "medium") -> None:
        import anthropic

        self.client = anthropic.Anthropic()  # SDK retries 429/5xx twice by default
        self.model, self.effort = model, effort

    def generate(self, question: str, chunks: list[ScoredChunk]) -> str:
        """Call client.messages.create and return the answer text.

        - system = load_prompt("answer_system"); user = build_user_message(...)
        - effort goes in output_config={"effort": self.effort}
          (Opus 5.5 has no temperature parameter — effort is the main knob)
        - response.content is a list of blocks; keep only type == "text"
        - if response.stop_reason == "refusal", raise RefusalError
        - also check stop_reason == "max_tokens": what does it mean for the answer?
        - log the exact prompt and the usage (input/output tokens): you'll want them for cost
        Bonus: set an explicit timeout on the client.
        """
        raise NotImplementedError("exercise 5c")


class BedrockLLM(ClaudeLLM):
    """Same Messages API, through Amazon Bedrock. Once 5c works, this works too."""

    def __init__(self, model: str, region: str, effort: str = "medium") -> None:
        from anthropic import AnthropicBedrockMantle

        self.client = AnthropicBedrockMantle(aws_region=region)
        self.model = model if model.startswith("anthropic.") else f"anthropic.{model}"
        self.effort = effort


def make_llm(kind: str, *, model: str, effort: str, region: str) -> LLM:
    if kind == "extractive":
        return ExtractiveLLM()
    if kind == "claude":
        return ClaudeLLM(model, effort)
    if kind == "bedrock":
        return BedrockLLM(model, region, effort)
    raise ValueError(f"unknown llm: {kind}")
