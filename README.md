# study-rag

A retrieval-augmented study assistant for my master's courses. Point it at lecture
notes, slides and papers, ask questions, and get answers **grounded in the course
material with citations** — not the model's general knowledge.

Built to be measured: every change to chunking, embeddings or retrieval is scored
against a golden question set in CI.

```text
 data/<course>/*.md|pdf ──► loader ──► structure-aware chunker ──► embedder (BGE)
                                                                      │
                                                         ┌────────────┴────────────┐
                                                         ▼                         ▼
                                                  vector index (HNSW)      full-text index (BM25)
                                                         └──────► RRF fusion ◄─────┘
                                                                     │
                                                       cross-encoder rerank (optional)
                                                                     │
                                          Claude (API or Amazon Bedrock) ──► cited answer
```

## Design decisions

| Decision | Why |
|---|---|
| Heading-aware chunking | Course notes are organised by topic; one chunk ≈ one concept, and the heading path makes citations readable. |
| Hybrid search + RRF | Embeddings catch paraphrases, BM25 catches exact terms (acronyms, formula names). RRF fuses *ranks*, so no score calibration. |
| Cross-encoder rerank on top-N only | Precision where it matters at bounded cost. |
| Pluggable backends via config | Same code runs offline at $0 (CI) and on AWS (Bedrock + pgvector). |
| Golden set labelled by `(source, phrase)` | Stays valid when chunk size changes, so experiments are comparable. |

## Quick start (local, $0)

```bash
uv venv && uv pip install -e ".[dev]"
study-rag ingest data/sample
study-rag ask "Why is ReLU preferred over sigmoid?"
study-rag eval eval/golden.jsonl
```

With real models (downloads ~150 MB once, runs on CPU):

```bash
uv pip install -e ".[local]"
export STUDY_RAG_EMBEDDER=hf STUDY_RAG_RERANKER=hf
```

With Postgres + pgvector:

```bash
docker compose up -d
uv pip install -e ".[pg]"
export STUDY_RAG_STORE=pgvector
```

With Claude generating the answers:

```bash
uv pip install -e ".[claude]"
export STUDY_RAG_LLM=claude          # ANTHROPIC_API_KEY or `ant auth login`
# or
export STUDY_RAG_LLM=bedrock AWS_REGION=us-east-1   # AWS credentials
```

## Evaluation

`study-rag eval` reports recall@1, recall@k and MRR for vector-only, keyword-only and
hybrid retrieval, so the value of each component is visible. CI fails if hybrid
recall@5 drops below the threshold.

<!-- Paste the latest eval table here -->

## Using your own courses

Put files under `data/<course-name>/` (`.md`, `.txt`, `.pdf`), then add 20–50
questions you actually had while studying to `eval/golden.jsonl`.

## Build log

The core components are written by hand. No LangChain or LlamaIndex: the point is to understand each piece.

| # | Component | File | Tests |
|---|---|---|---|
| 1 | Retrieval metrics (recall@k, MRR) | `eval/metrics.py` | `test_metrics.py` |
| 2 | Chunking | `ingest/chunking.py` | `test_chunking.py` |
| 3 | Vector search + BM25 | `store/memory.py` | `test_store.py` |
| 4 | Reciprocal Rank Fusion | `retrieval.py` | `test_store.py` |
| 5 | Prompt + Claude call | `prompts/`, `llm.py` | `test_pipeline.py` |

## Roadmap

- [ ] LLM-as-judge for answer faithfulness (are claims supported by the cited excerpt?)
- [ ] Terraform: S3 + Lambda + Bedrock with least-privilege IAM, budget alarm
- [ ] GitHub Actions → AWS via OIDC (no long-lived keys)
- [ ] CloudWatch metrics: latency, tokens, retrieval scores
- [ ] Bedrock Guardrails
