# Spec 001 — Core RAG over course materials

## Problem
While studying, answers are scattered across weeks of slides and notes. General
chatbots answer from general knowledge, not from *what my course taught*, and
can't point me to the lecture to re-read.

## User stories
1. As a student, I ingest a folder per course (`data/<course>/`) of md/txt/pdf files.
2. As a student, I ask a question and get a concise answer with `[n]` citations
   that map to file + section (+ page for PDFs).
3. If my materials don't cover the question, the assistant says so rather than guessing.
4. As the developer, I can see whether a change improved retrieval before merging.

## Acceptance criteria
- [x] Ingest md/txt/pdf; course name derived from folder.
- [ ] Chunks carry heading path; size/overlap configurable.
- [ ] Hybrid retrieval (vector + BM25, RRF) with optional cross-encoder rerank.
- [ ] Generation via Claude API or Bedrock; offline extractive mode for CI.
- [ ] Eval reports recall@1, recall@k, MRR per retrieval mode; CI gate on hybrid recall@5 ≥ 0.8.
- [ ] Answer faithfulness eval (LLM-as-judge) on a small sample, run manually (costs tokens).
- [x] Per-course filter (`--course machine-learning`).
- [x] Per-query JSONL traces: retrieved chunks, scores, stage timings, token usage.
- [x] Embedding cache keyed by (model, text).
- [x] HTTP API; deployable to Lambda behind an IAM-authenticated Function URL.
- [x] Index records how it was built; mismatched embedder config fails at startup.
- [x] Index persisted without pickle; atomic writes.
- [x] Cost per query (USD) in traces and CloudWatch; 24h cost alarm.
- [x] Feedback endpoint; negative feedback exported as golden-set candidates.
- [x] Streaming endpoint (SSE).
- [x] Upload to S3 triggers reindex; warm instances reload the new index.
- [ ] Streaming generation with Claude (exercise 5d).
- [ ] Prompt-injection test + hardened prompt (exercise 8).

## Non-goals (for now)
- Multi-user auth, web UI, chat history.
- Managed vector DB (OpenSearch, Aurora): cost at idle isn't justified at this corpus size.

## Constraints
- Default path must run at $0 with no network access.
- No AWS resource without a budget alarm; no long-lived AWS keys in CI.
