# Tasks 003 — Regression gate

- [x] `eval/baseline.py`: `save_baseline`, `load_baseline`, `regressions`
- [x] CLI options `--baseline`, `--max-drop`, `--save-baseline`
- [x] `tests/test_baseline.py` (CLI round trip included), in the CI `plumbing` job
- [x] CI `retrieval-eval` uses the baseline when present; `make baseline`, `make eval`
- [ ] **Owner:** after exercises 1–4, run `make baseline` and commit `eval/baseline.json`
- [ ] **Owner:** exercise 9: measure noise, then set `--max-drop` from data
