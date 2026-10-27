"""Infrastructure code — independent of the exercises, should always pass."""

import json

import numpy as np
from fastapi.testclient import TestClient

from study_rag import api
from study_rag.cache import CachedEmbedder
from study_rag.embeddings import HashEmbedder
from study_rag.models import Answer, Chunk, ScoredChunk
from study_rag.retrieval import course_of
from study_rag.tracing import Trace, write_trace


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def embed(self, texts):
        self.calls.append(len(texts))
        return super().embed(texts)


def test_cache_embeds_each_text_once_and_persists(tmp_path):
    inner = CountingEmbedder()
    cache = CachedEmbedder(inner, "m", str(tmp_path / "c.pkl"))
    first = cache.embed(["a", "b"])
    second = cache.embed(["b", "c"])
    assert inner.calls == [2, 1]  # only "c" was new, and misses are batched
    assert np.allclose(first[1], second[0])
    cache.save()

    reloaded = CachedEmbedder(CountingEmbedder(), "m", str(tmp_path / "c.pkl"))
    reloaded.embed(["a", "b", "c"])
    assert reloaded.inner.calls == []


def test_cache_key_includes_model(tmp_path):
    inner = CountingEmbedder()
    CachedEmbedder(inner, "m1", str(tmp_path / "c.pkl")).embed(["a"])
    CachedEmbedder(inner, "m2", str(tmp_path / "c.pkl")).embed(["a"])
    assert inner.calls == [1, 1]


def test_course_of():
    def sc(src):
        return ScoredChunk(Chunk("id", "d", "t", src), 0.0)

    assert course_of(sc("machine-learning/week1.md")) == "machine-learning"
    assert course_of(sc("loose.md")) == ""


def test_trace_written_as_jsonl(tmp_path):
    t = Trace(question="q")
    with t.timed("retrieve"):
        pass
    t.record_retrieved([ScoredChunk(Chunk("c1", "d", "t", "a.md", "S"), 0.5)])
    path = tmp_path / "t.jsonl"
    write_trace(t, str(path))
    write_trace(t, str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["retrieved"][0]["chunk_id"] == "c1"
    assert "retrieve" in rows[0]["timings_ms"]


class FakeRAG:
    class store:  # noqa: N801
        meta = {"embedder": "hash"}

        def __len__(self):
            return 3

    store = store()
    prompt_version = "abc"

    def ask(self, question, course=None):
        src = ScoredChunk(Chunk("c1", "d", "t", "ml/w1.md", "Bias"), 0.9)
        return Answer(text=f"answer to {question} [1]", sources=[src], model="fake")


def test_api_ask_and_health(monkeypatch):
    monkeypatch.setattr(api, "get_rag", lambda: FakeRAG())
    client = TestClient(api.app)
    assert client.get("/health").json()["chunks"] == 3
    r = client.post("/ask", json={"question": "what is bias?"})
    assert r.status_code == 200
    assert r.json()["sources"][0] == {"n": 1, "source": "ml/w1.md", "section": "Bias", "score": 0.9}


def test_api_validates_input(monkeypatch):
    monkeypatch.setattr(api, "get_rag", lambda: FakeRAG())
    assert TestClient(api.app).post("/ask", json={"question": ""}).status_code == 422


def test_cli_ask_prints_answer_and_sources(monkeypatch):
    from typer.testing import CliRunner

    from study_rag import cli

    src = ScoredChunk(Chunk("c1", "d", "t", "ml/week1.md", "Dropout"), 0.42)

    class FakeRAG:
        def ask(self, question, course=None):
            return Answer(f"{question}|{course}", [src], "fake", trace_id="t1")

    monkeypatch.setattr(cli, "_rag", FakeRAG)
    result = CliRunner().invoke(cli.app, ["ask", "what?", "--course", "ml"])
    assert result.exit_code == 0
    assert "what?|ml" in result.output
    assert "[1] ml/week1.md - Dropout  (score 0.420)" in result.output


def test_cli_eval_gate_fails_below_min_recall(monkeypatch):
    from typer.testing import CliRunner

    from study_rag import cli

    monkeypatch.setattr(cli, "_rag", lambda: None)
    monkeypatch.setattr(cli, "load_golden", lambda path: [])
    monkeypatch.setattr(cli, "evaluate", lambda rag, golden, k: {"hybrid": {f"recall@{k}": 0.5}})
    monkeypatch.setattr(cli, "format_report", lambda report: "report")
    args = ["eval", "g.jsonl", "-k", "3", "--min-recall"]
    runner = CliRunner()
    assert runner.invoke(cli.app, [*args, "0.4"]).exit_code == 0
    failed = runner.invoke(cli.app, [*args, "0.8"])
    assert failed.exit_code == 1
    assert "FAIL: hybrid recall@3=0.500 < 0.8" in failed.output


def test_settings_read_env_by_prefix_and_alias(monkeypatch):
    from study_rag.config import Settings

    monkeypatch.setenv("STUDY_RAG_TOP_K", "9")  # prefix + field name
    monkeypatch.setenv("STUDY_RAG_INDEX", "/tmp/x.npz")  # explicit alias
    monkeypatch.setenv("AWS_REGION", "eu-west-1")  # alias outside the prefix
    s = Settings()
    assert (s.top_k, s.index_path, s.aws_region) == (9, "/tmp/x.npz", "eu-west-1")
    assert Settings(index_path="/tmp/y.npz").index_path == "/tmp/y.npz"  # kwargs beat env


def test_settings_reject_bad_values_at_startup(monkeypatch):
    import pytest
    from pydantic import ValidationError

    from study_rag.config import Settings

    monkeypatch.setenv("STUDY_RAG_LLM", "claud")  # typo
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.delenv("STUDY_RAG_LLM")
    monkeypatch.setenv("STUDY_RAG_TOP_K", "0")
    with pytest.raises(ValidationError):
        Settings()
