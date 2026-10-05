"""Production plumbing: index format, compatibility checks, cost, metrics, feedback,
streaming API, S3 reindexing. Independent of the exercises — must always pass."""

import json

import boto3
import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from study_rag import api, pipeline
from study_rag.config import Settings
from study_rag.embeddings import HashEmbedder
from study_rag.eval.feedback import negative_feedback_candidates
from study_rag.models import Chunk, Document, ScoredChunk
from study_rag.pricing import cost_usd
from study_rag.s3sync import IndexWatcher, reindex
from study_rag.store.base import IndexMismatchError, check_compatible
from study_rag.store.memory import MemoryStore
from study_rag.tracing import Trace, write_trace


def one_chunk_per_doc(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Stand-in for exercise 2 so plumbing tests don't depend on it."""
    return [Chunk(f"{doc.doc_id}:0", doc.doc_id, doc.text, doc.source)]


@pytest.fixture
def stub_chunker(monkeypatch):
    monkeypatch.setattr(pipeline, "chunk_document", one_chunk_per_doc)


# --- index format ---------------------------------------------------------------
def test_index_roundtrip_without_pickle(tmp_path):
    path = str(tmp_path / "index.npz")
    store = MemoryStore(path)
    chunks = [Chunk("a:0", "a", "text one", "c/a.md", "Sec", 0), Chunk("b:0", "b", "two", "b.md")]
    vecs = HashEmbedder().embed([c.text for c in chunks])
    store.add(chunks, vecs)
    store.meta = {"embedder": "hash", "dim": 512}
    store.save()

    with np.load(path, allow_pickle=False) as data:  # would raise if pickled objects inside
        assert data["vectors"].shape == (2, 512)

    loaded = MemoryStore.load(path)
    assert loaded.chunks == chunks
    assert np.allclose(loaded.vectors, vecs)
    assert loaded.meta == {"embedder": "hash", "dim": 512}


def test_empty_index_roundtrip(tmp_path):
    store = MemoryStore(str(tmp_path / "i.npz"))
    store.meta = {"dim": 512}
    store.save()
    assert len(MemoryStore.load(str(tmp_path / "i.npz"))) == 0


# --- compatibility --------------------------------------------------------------
def test_check_compatible():
    meta = {"embedder": "hf", "embed_model": "bge", "dim": 384, "chunk_size": 800}
    check_compatible(meta, {**meta, "chunk_size": 300})  # chunking may differ
    with pytest.raises(IndexMismatchError, match="embed_model"):
        check_compatible(meta, {**meta, "embed_model": "other"})
    with pytest.raises(IndexMismatchError, match="no metadata"):
        check_compatible({}, meta)


def test_pipeline_refuses_index_from_another_model(tmp_path):
    path = str(tmp_path / "i.npz")
    store = MemoryStore(path)
    store.add([Chunk("a:0", "a", "t", "a.md")], np.ones((1, 384), np.float32))
    store.meta = {"embedder": "hf", "embed_model": "BAAI/bge-small-en-v1.5", "dim": 384}
    store.save()
    with pytest.raises(IndexMismatchError):
        pipeline.StudyRAG.from_settings(Settings(index_path=path))  # config: hash embedder


def test_ingest_records_metadata(tmp_path, stub_chunker):
    (tmp_path / "docs" / "ml").mkdir(parents=True)
    (tmp_path / "docs" / "ml" / "w1.md").write_text("bias variance")
    rag = pipeline.StudyRAG.from_settings(Settings(index_path=str(tmp_path / "i.npz")))
    rag.ingest(tmp_path / "docs")
    meta = MemoryStore.load(str(tmp_path / "i.npz")).meta
    assert meta["embedder"] == "hash" and meta["dim"] == 512 and meta["format_version"] == 1


# --- cost + metrics -------------------------------------------------------------
def test_cost():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(4.0 + 2.0)
    assert cost_usd("anthropic.claude-opus-5-5", usage) == pytest.approx(6.0)
    assert cost_usd("unknown-model", usage) is None
    assert cost_usd("claude-opus-5-5", None) is None


def test_stdout_trace_carries_cloudwatch_emf(capsys):
    t = Trace(question="q", model="claude-opus-5-5", cost_usd=0.01,
              usage={"input_tokens": 10, "output_tokens": 5})
    t.timings_ms = {"retrieve": 3.0, "generate": 900.0}
    write_trace(t, "-")
    rec = json.loads(capsys.readouterr().out)
    names = {m["Name"] for m in rec["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
    assert {"RetrieveMs", "GenerateMs", "CostUSD", "OutputTokens", "Errors"} <= names
    assert rec["CostUSD"] == 0.01 and rec["Model"] == "claude-opus-5-5"
    assert rec["type"] == "query_trace" and rec["question"] == "q"


def test_file_trace_has_no_emf(tmp_path):
    write_trace(Trace(question="q"), str(tmp_path / "t.jsonl"))
    assert "_aws" not in json.loads((tmp_path / "t.jsonl").read_text())


# --- feedback loop --------------------------------------------------------------
def test_negative_feedback_becomes_golden_candidate():
    events = [
        {"type": "query_trace", "trace_id": "t1", "question": "q1", "answer": "a",
         "retrieved": [{"source": "x.md", "section": "S"}], "prompt_version": "p"},
        {"type": "query_trace", "trace_id": "t2", "question": "q2", "retrieved": []},
        {"type": "feedback", "trace_id": "t1", "rating": -1, "comment": "wrong"},
        {"type": "feedback", "trace_id": "t2", "rating": 1},
    ]
    (cand,) = negative_feedback_candidates(events)
    assert cand["question"] == "q1" and cand["retrieved"] == ["x.md — S"]
    assert cand["source"] == "TODO"


# --- API ------------------------------------------------------------------------
class StreamingFakeRAG:
    def __init__(self):
        self.feedback = []

    def ask_stream(self, question, course=None):
        src = [ScoredChunk(Chunk("c1", "d", "t", "ml/w1.md", "Bias"), 0.5)]
        return "trace123", src, iter(["Hello ", "world [1]"])

    def record_feedback(self, trace_id, rating, comment=None):
        self.feedback.append((trace_id, rating, comment))


def parse_sse(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_endpoint_emits_sources_deltas_done(monkeypatch):
    monkeypatch.setattr(api, "get_rag", lambda: StreamingFakeRAG())
    r = TestClient(api.app).post("/ask/stream", json={"question": "what is bias?"})
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["sources", "delta", "delta", "done"]
    assert events[0][1][0]["source"] == "ml/w1.md"
    assert "".join(d["text"] for e, d in events if e == "delta") == "Hello world [1]"
    assert events[-1][1] == {"trace_id": "trace123"}


def test_feedback_endpoint(monkeypatch):
    fake = StreamingFakeRAG()
    monkeypatch.setattr(api, "get_rag", lambda: fake)
    client = TestClient(api.app)
    assert client.post("/feedback", json={"trace_id": "trace123", "rating": -1}).status_code == 204
    assert fake.feedback == [("trace123", -1, None)]
    assert client.post("/feedback", json={"trace_id": "trace123", "rating": 5}).status_code == 422


def test_events_ignores_non_s3_and_reindexes_on_s3(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "reindex", lambda s: calls.append(1) or {"chunks": 3})
    client = TestClient(api.app)
    assert client.post("/events", json={"source": "aws.events"}).json() == {"ignored": True}
    s3_event = {"Records": [{"eventSource": "aws:s3", "s3": {"object": {"key": "docs/a.md"}}}]}
    assert client.post("/events", json=s3_event).json()["reindexed"] == {"chunks": 3}
    assert calls == [1]


# --- S3 reindexing --------------------------------------------------------------
@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="rag-test")
        yield s3


def test_reindex_from_s3_publishes_loadable_index(aws, tmp_path, stub_chunker):
    aws.put_object(Bucket="rag-test", Key="docs/ml/w1.md", Body=b"dropout regularises")
    aws.put_object(Bucket="rag-test", Key="docs/cloud/l3.md", Body=b"cap theorem")
    settings = Settings(
        index_path=str(tmp_path / "i.npz"),
        index_s3_uri="s3://rag-test/index/index.npz",
        docs_s3_uri="s3://rag-test/docs/",
    )
    assert reindex(settings) == {"files": 2, "chunks": 2}

    aws.download_file("rag-test", "index/index.npz", str(tmp_path / "dl.npz"))
    store = MemoryStore.load(str(tmp_path / "dl.npz"))
    assert sorted(c.source for c in store.chunks) == ["cloud/l3.md", "ml/w1.md"]

    # A fresh process picks the published index up from S3.
    rag = pipeline.StudyRAG.from_settings(
        Settings(index_path=str(tmp_path / "fresh.npz"), index_s3_uri="s3://rag-test/index/index.npz")
    )
    assert len(rag.store) == 2


def test_missing_index_in_s3_means_empty_store(aws, tmp_path):
    rag = pipeline.StudyRAG.from_settings(
        Settings(index_path=str(tmp_path / "i.npz"), index_s3_uri="s3://rag-test/index/index.npz")
    )
    assert len(rag.store) == 0


def test_index_watcher_detects_new_version(aws):
    aws.put_object(Bucket="rag-test", Key="index/index.npz", Body=b"v1")
    w = IndexWatcher("s3://rag-test/index/index.npz", interval=0)
    w.mark_loaded()
    assert not w.changed()
    aws.put_object(Bucket="rag-test", Key="index/index.npz", Body=b"v2")
    assert w.changed()
