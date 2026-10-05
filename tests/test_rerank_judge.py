"""Exercises 6 and 7 — with fake models, so no download and no API key needed."""

from types import SimpleNamespace

import pytest

from study_rag.eval.judge import Verdict, judge_answer
from study_rag.llm import RefusalError
from study_rag.models import Chunk, ScoredChunk
from study_rag.retrieval import CrossEncoderReranker


def sc(cid: str, text: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(Chunk(cid, "d", text, "s.md"), score)


# --- 6 ------------------------------------------------------------------------
class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the passage."""

    def __init__(self):
        self.seen = []

    def predict(self, pairs):
        self.seen = list(pairs)
        return [sum(w in p.lower() for w in q.lower().split()) for q, p in pairs]


def make_reranker() -> CrossEncoderReranker:
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)  # skip model download
    r.model = FakeCrossEncoder()
    return r


def test_rerank_reorders_by_cross_encoder_score():
    r = make_reranker()
    cands = [sc("a", "nothing here", 0.9), sc("b", "dropout prevents overfitting", 0.1)]
    out = r.rerank("dropout overfitting", cands)
    assert [c.chunk.chunk_id for c in out] == ["b", "a"]
    assert out[0].score == 2
    assert r.model.seen[0] == ("dropout overfitting", "nothing here")  # (query, passage)


def test_rerank_empty():
    assert make_reranker().rerank("q", []) == []


# --- 7 ------------------------------------------------------------------------
class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.kwargs = {}

    def parse(self, **kwargs):
        self.kwargs = kwargs
        verdict = Verdict(unsupported_claims=["SVMs use kernels"], abstained=False, faithful=False)
        return SimpleNamespace(
            parsed_output=verdict, stop_reason=self.stop_reason, stop_details=None,
            content=[], usage=None,
        )


def test_judge_uses_structured_output_and_sends_answer_and_excerpts():
    msgs = FakeMessages()
    client = SimpleNamespace(messages=msgs)
    v = judge_answer(client, "m", "q?", "the answer under test", [sc("a", "excerpt text")])
    assert isinstance(v, Verdict) and v.unsupported_claims == ["SVMs use kernels"]
    assert msgs.kwargs["output_format"] is Verdict
    sent = str(msgs.kwargs["messages"])
    assert "the answer under test" in sent and "excerpt text" in sent
    assert msgs.kwargs.get("system")


def test_judge_raises_on_refusal():
    client = SimpleNamespace(messages=FakeMessages(stop_reason="refusal"))
    with pytest.raises(RefusalError):
        judge_answer(client, "m", "q?", "a", [sc("a", "t")])
