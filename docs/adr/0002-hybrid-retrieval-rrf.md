# 0002. Hybrid retrieval fused with Reciprocal Rank Fusion

**Status:** accepted

## Context
Embeddings catch paraphrases but blur exact terms (acronyms, course codes, formula names).
Keyword search has the opposite profile. Cosine scores live in [-1, 1]; BM25 scores are
unbounded.

## Decision
Run vector search and BM25, fuse the two *rankings* with RRF (`1/(k+rank)`, k=60), optionally
rerank the top-N with a cross-encoder. Retrieval modes (`vector`, `keyword`, `hybrid`) are
all evaluated so the report shows what hybrid buys.

## Alternatives rejected
- **Weighted score sum:** needs score normalisation and a tuned weight per corpus.
- **Vector only:** loses exact-term queries.

## Consequences
- No calibration between score scales.
- RRF discards score magnitude: a very confident match and a barely-relevant one at the same
  rank count the same. The reranker is where precision is recovered.
- Whether hybrid actually wins on this corpus is an empirical question: see the README results.
