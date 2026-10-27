# Spec 003 — Retrieval regression gate

**Status:** implemented; baseline file pending (needs exercises 1–4 first).

## Problem
CI fails only when hybrid recall@5 is under 0.8. A change that drops it from 0.95 to 0.82
passes. Quality should be gated relative to the last accepted result.

## Acceptance criteria
- [x] `study-rag eval --save-baseline PATH` writes the report as JSON.
- [x] `study-rag eval --baseline PATH [--max-drop 0.03]` exits 1 and names each hybrid metric
      that fell by more than the tolerance.
- [x] A metric missing from the run (different `-k`) is a regression, not a silent pass.
- [x] CI uses the baseline only when `eval/baseline.json` exists; `make baseline` refreshes it.
- [ ] `eval/baseline.json` committed (after exercises 1–4 and a real golden set).
- [ ] Tolerance justified by measured run-to-run noise (exercise 9).

## Notes
The fixed `--min-recall` floor stays as a second, absolute guard. See ADR 0010: with 30
questions one question is 0.033, so the default 0.03 tolerance is about one question.
