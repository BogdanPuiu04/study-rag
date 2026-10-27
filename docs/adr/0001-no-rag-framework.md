# 0001. No RAG framework: core built on numpy and the Anthropic SDK

**Status:** accepted

## Context
The system has to be explainable and measurable step by step. Frameworks (LangChain,
LlamaIndex) hide chunking, fusion, reranking and prompt assembly behind abstractions that
change between releases.

## Decision
Chunking, vector search, BM25, rank fusion, reranking, the grounded prompt and the judge are
implemented directly on numpy and the Anthropic SDK, each behind a small `Protocol` and
covered by its own tests.

## Alternatives rejected
- **LangChain / LlamaIndex:** faster to a demo, but every choice becomes "the framework does
  that", and the dependency surface is large for a Lambda image.

## Consequences
- Every retrieval decision can be explained and measured.
- More code to own; no ecosystem integrations for free.
