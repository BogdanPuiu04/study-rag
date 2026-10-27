# 0007. EMF metrics and an app-level cost alarm

**Status:** accepted

## Context
"What does an answer cost?" and "did latency regress?" need numbers per query, without extra
API calls or latency. AWS Budgets lag by up to a day.

## Decision
Each query writes one JSON trace line to stdout; CloudWatch extracts metrics from it (EMF:
latency per stage, tokens, cost in USD). An alarm on `CostUSD` reacts within the hour; the
AWS Budget remains as the backstop. The trace carries `prompt_version`, retrieved chunks and
scores, so a bad answer is debugged from the trace.

## Consequences
- Metrics and the record that produced them are the same line.
- Token prices are a table in `pricing.py`, dated, and unknown models return `None` instead
  of a guessed cost.
- Bedrock prices differ from API list prices: override with `STUDY_RAG_PRICE_IN/OUT`.
