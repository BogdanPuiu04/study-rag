"""End-to-end: passes once exercises 1-4 and 5b are done (no API calls)."""

from pathlib import Path

import pytest

from study_rag.config import Settings
from study_rag.eval.run import evaluate, load_golden
from study_rag.llm import ExtractiveLLM, build_user_message, cited_indices, load_prompt
from study_rag.pipeline import StudyRAG

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    settings = Settings(index_path=str(tmp_path_factory.mktemp("idx") / "i.pkl"))
    rag = StudyRAG.from_settings(settings)
    rag._llm = ExtractiveLLM()
    rag.ingest(ROOT / "data" / "sample")
    return rag


def test_ask_returns_cited_answer_with_sources(rag):
    answer = rag.ask("What does dropout do?")
    assert answer.sources
    assert cited_indices(answer.text) <= set(range(1, len(answer.sources) + 1))
    assert "co-adapting" in answer.text


def test_user_message_numbers_excerpts_and_puts_question_last(rag):
    chunks = rag.retriever().retrieve("dropout", 2)
    msg = build_user_message("Q?", chunks)
    assert "1" in msg and "2" in msg
    assert chunks[0].chunk.source in msg
    assert msg.rstrip().endswith("Q?")


def test_system_prompt_written():  # 5a
    assert len(load_prompt("answer_system")) > 50


def test_hybrid_retrieval_meets_quality_bar(rag):
    report = evaluate(rag, load_golden(ROOT / "eval" / "golden.jsonl"), k=5)
    assert report["hybrid"]["recall@5"] >= 0.8
