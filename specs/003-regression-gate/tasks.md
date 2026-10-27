# Tasks 003 — Regression gate

- [x] `eval/baseline.py`: `save_baseline`, `load_baseline`, `regressions`
- [x] CLI options `--baseline`, `--max-drop`, `--save-baseline`
- [x] `tests/test_baseline.py` (CLI round trip included), in the CI `plumbing` job
- [x] CI `retrieval-eval` uses the baseline when present; `make baseline`, `make eval`
- [ ] With a real golden set, run `make baseline` and commit `eval/baseline.json`
- [ ] Measure run-to-run noise, then set `--max-drop` from data
