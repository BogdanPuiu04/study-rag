# 0010. Retrieval regression gate against a baseline

**Status:** accepted

## Context
CI gated retrieval on a fixed floor (hybrid recall@5 >= 0.8). A floor passes while quality
erodes toward it.

## Decision
`study-rag eval --baseline eval/baseline.json --max-drop 0.03` fails when any hybrid metric
falls more than 0.03 (absolute) below the saved report. `make baseline` accepts the current
numbers as the new reference; the file is committed in the PR that moved them, so history
shows what each change did. CI applies the baseline only once the file exists. A metric
missing from the run (e.g. a different `-k`) counts as a regression rather than passing silently.

## Consequences
- The golden set must be big enough that 0.03 is above noise: with 30 questions, one
  question is 0.033. Measure run-to-run variation before trusting a threshold (exercise 9).
- Improvements don't fail; they only matter once someone refreshes the baseline.
