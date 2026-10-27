"""Answer generation. Three backends behind one interface:

- extractive: $0, offline — returns the best-matching passages verbatim. Used in CI.
- claude:     Claude API (ANTHROPIC_API_KEY or `ant auth login`).
- bedrock:    Claude on Amazon Bedrock (AWS credentials; IAM role in AWS).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from study_rag.models import ScoredChunk

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def prompt_version(name: str) -> str:
    """Short content hash of a prompt file, recorded in every trace, so a quality
    change can be tied to the exact prompt that produced it."""
    try:
        return hashlib.sha256(load_prompt(name).encode()).hexdigest()[:12]
    except FileNotFoundError:
        return "missing"


def cited_indices(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


# Excerpts are untrusted text pasted into the prompt. A document containing "</excerpt>" or
# "<question>" must not be able to close the block early and smuggle in text that reads as
# coming from us, so tag-like sequences are neutralised before wrapping.
_TAG_LIKE = re.compile(r"<(/?)(excerpt|question|answer)\b", re.IGNORECASE)


def escape_tags(text: str) -> str:
    return _TAG_LIKE.sub(r"&lt;\1\2", text)


def format_excerpts(chunks: list[ScoredChunk]) -> str:
    """Numbered (1-based) excerpts in XML-style tags, each with its provenance, so `[n]`
    citations map back to a chunk and the model can tell data from instructions."""
    parts = []
    for i, sc in enumerate(chunks, start=1):
        c = sc.chunk
        where = f"{c.source} - {c.section}" if c.section else c.source
        where = escape_tags(where).replace('"', "&quot;")
        parts.append(f'<excerpt n="{i}" source="{where}">\n{escape_tags(c.text)}\n</excerpt>')
    return "\n\n".join(parts)


def build_user_message(question: str, chunks: list[ScoredChunk]) -> str:
    """The user turn: the numbered excerpts, then the question.

    The question goes last: with long context, an instruction placed after the documents is
    followed more reliably than one placed before them.
    """
    excerpts = format_excerpts(chunks) if chunks else "(no excerpts were retrieved)"
    return f"{excerpts}\n\nQuestion: {question}"


class LLM(Protocol):
    model: str
    # Token usage of the most recent call ({"input_tokens": .., "output_tokens": ..}),
    # or None. The pipeline turns it into cost in the trace.
    last_usage: dict[str, int] | None

    def generate(self, question: str, chunks: list[ScoredChunk]) -> str: ...

    def generate_stream(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]:
        """Yield the answer in text pieces as they're produced."""
        ...


class ExtractiveLLM:
    """No model call: stitches the top passages together with citations."""

    model = "extractive"
    last_usage: dict[str, int] | None = None

    def generate(self, question: str, chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return "I couldn't find anything about that in your course materials."
        return "\n\n".join(f"{sc.chunk.text.strip()} [{i}]" for i, sc in enumerate(chunks[:2], 1))

    def generate_stream(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]:
        for paragraph in self.generate(question, chunks).split("\n\n"):
            yield paragraph + "\n\n"


class RefusalError(RuntimeError):
    pass


class ClaudeLLM:
    # Thinking tokens count against max_tokens, so leave room beyond the visible answer.
    max_tokens = 16000
    # Server-side refusal fallback: if a safety classifier declines, the API re-runs the
    # request on a fallback model inside the same call. Claude API only; Bedrock has no
    # `fallbacks` parameter (see BedrockLLM).
    use_fallbacks = True
    TRUNCATED_NOTE = "\n\n[The answer was cut off at the length limit.]"

    def __init__(self, model: str, effort: str = "medium") -> None:
        import anthropic

        # An explicit timeout instead of the SDK's 10 minutes: a stuck answer should fail
        # fast. The SDK still retries 429/5xx/connection errors twice with backoff.
        self.client = anthropic.Anthropic(timeout=120.0)
        self.model, self.effort = model, effort
        self.last_usage: dict[str, int] | None = None

    def _request(self, question: str, chunks: list[ScoredChunk]) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": load_prompt("answer_system"),
            "messages": [{"role": "user", "content": build_user_message(question, chunks)}],
            # Effort is the quality/cost knob on current models (there is no temperature).
            "output_config": {"effort": self.effort},
        }
        if self.use_fallbacks:
            params |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return params

    def _messages(self):
        return self.client.beta.messages if self.use_fallbacks else self.client.messages

    def _finish(self, response) -> None:
        """Record usage and check why generation stopped. Shared by both call styles."""
        usage = response.usage
        self.last_usage = (
            {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
            if usage is not None
            else None
        )
        if response.stop_reason == "refusal":
            raise RefusalError("the model declined to answer this question")

    def generate(self, question: str, chunks: list[ScoredChunk]) -> str:
        response = self._messages().create(**self._request(question, chunks))
        self._finish(response)
        # Content is a list of blocks (thinking, fallback markers, text): keep the text.
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "max_tokens":
            # A truncated answer is still useful, but the reader must know it is incomplete.
            text += self.TRUNCATED_NOTE
        return text

    def generate_stream(self, question: str, chunks: list[ScoredChunk]) -> Iterator[str]:
        """Same request, yielded as it is generated: time to first token drops from the
        whole generation time to well under a second, which is what users perceive.
        Usage and the stop reason are only known at the end, so they are checked there."""
        with self._messages().stream(**self._request(question, chunks)) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        self._finish(final)
        if final.stop_reason == "max_tokens":
            yield self.TRUNCATED_NOTE


class BedrockLLM(ClaudeLLM):
    """Same Messages API, through Amazon Bedrock. Once 5c works, this works too."""

    def __init__(self, model: str, region: str, effort: str = "medium") -> None:
        from anthropic import AnthropicBedrockMantle

        # Same Messages API as anthropic.Anthropic, different class: duck-typed on purpose.
        self.client = AnthropicBedrockMantle(aws_region=region)  # type: ignore[assignment]
        self.model = model if model.startswith("anthropic.") else f"anthropic.{model}"
        self.effort = effort
        self.last_usage = None
        self.use_fallbacks = False


def make_llm(kind: str, *, model: str, effort: str, region: str) -> LLM:
    if kind == "extractive":
        return ExtractiveLLM()
    if kind == "claude":
        return ClaudeLLM(model, effort)
    if kind == "bedrock":
        return BedrockLLM(model, region, effort)
    raise ValueError(f"unknown llm: {kind}")
