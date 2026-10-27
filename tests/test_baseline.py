"""Regression gate against a saved eval report."""

from typer.testing import CliRunner

from study_rag import cli
from study_rag.eval.baseline import load_baseline, regressions, save_baseline

BASE = {"hybrid": {"recall@1": 0.70, "recall@5": 0.90, "mrr": 0.80}, "vector": {"mrr": 0.5}}


def report(**hybrid):
    return {"hybrid": {**BASE["hybrid"], **hybrid}}


def test_no_change_or_improvement_passes():
    assert regressions(report(), BASE) == []
    assert regressions(report(mrr=0.95, **{"recall@5": 1.0}), BASE) == []


def test_small_drop_within_tolerance_passes_and_large_drop_fails():
    assert regressions(report(mrr=0.78), BASE, max_drop=0.03) == []
    problems = regressions(report(mrr=0.70), BASE, max_drop=0.03)
    assert len(problems) == 1 and "mrr" in problems[0] and "0.800 -> 0.700" in problems[0]


def test_every_dropped_metric_is_reported():
    assert len(regressions(report(mrr=0.1, **{"recall@5": 0.1}), BASE)) == 2


def test_only_the_requested_mode_is_compared():
    # vector is not in the report at all: irrelevant, since the gate is on hybrid
    assert regressions(report(), BASE, mode="hybrid") == []


def test_missing_metric_is_a_regression_not_a_silent_pass():
    current = {"hybrid": {"recall@1": 0.7, "recall@3": 0.9, "mrr": 0.8}}  # run with -k 3
    problems = regressions(current, BASE)
    assert any("recall@5" in p and "missing" in p for p in problems)


def test_baseline_without_the_mode_is_reported():
    assert regressions(report(), {"vector": {"mrr": 0.5}}) == ["baseline has no 'hybrid' results"]


def test_roundtrip(tmp_path):
    path = tmp_path / "sub" / "baseline.json"
    save_baseline(BASE, path)
    assert load_baseline(path) == BASE


def test_cli_saves_then_gates_on_the_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_rag", lambda: None)
    monkeypatch.setattr(cli, "load_golden", lambda path: [])
    monkeypatch.setattr(cli, "format_report", lambda r: "report")
    current = {"report": BASE}
    monkeypatch.setattr(cli, "evaluate", lambda rag, golden, k: current["report"])
    runner, path = CliRunner(), str(tmp_path / "b.json")

    assert runner.invoke(cli.app, ["eval", "g.jsonl", "--save-baseline", path]).exit_code == 0
    assert runner.invoke(cli.app, ["eval", "g.jsonl", "--baseline", path]).exit_code == 0

    current["report"] = report(mrr=0.5)
    failed = runner.invoke(cli.app, ["eval", "g.jsonl", "--baseline", path])
    assert failed.exit_code == 1
    assert "regression" in failed.output and "mrr" in failed.output
