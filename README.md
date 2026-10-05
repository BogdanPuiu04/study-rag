# study-rag

A retrieval-augmented study assistant for my AI master's courses. Point it at
lecture notes, slides and papers, ask questions, and get answers **grounded in the
course material with citations**, not the model's general knowledge.

Built to be measured: every change to chunking, embeddings or retrieval is scored
against a golden question set, and answers are graded for faithfulness by an LLM judge.

```text
 data/<course>/*.md|pdf ──► loader ──► structure-aware chunker ──► embedder (BGE, cached)
                                                                      │
                                                         ┌────────────┴────────────┐
                                                         ▼                         ▼
                                                   vector search                 BM25
                                                         └──────► RRF fusion ◄─────┘
                                                                     │
                                                       cross-encoder rerank (optional)
                                                                     │
                                          Claude (API or Amazon Bedrock) ──► cited answer
                                                                     │
                                                          JSONL trace per query
```

## Results

<!-- Fill in as you go — this table is the point of the project. -->

| Configuration | recall@1 | recall@5 | MRR |
|---|---|---|---|
| BM25 only | | | |
| Vector only (hash baseline) | | | |
| Vector only (BGE-small) | | | |
| Hybrid (BGE + BM25, RRF) | | | |
| Hybrid + cross-encoder rerank | | | |

| Answer quality (LLM judge) | value |
|---|---|
| Faithfulness | |
| Abstention on unanswerable | |

## Design decisions

| Decision | Why |
|---|---|
| Chunking: *(your choice)* | *(your reasoning)* |
| Hybrid search + RRF | Embeddings catch paraphrases, BM25 catches exact terms (acronyms, formula names). RRF fuses *ranks*, so no score calibration. |
| Cross-encoder rerank on top-N only | Precision where it matters at bounded latency. |
| Golden set labelled by `(source, phrase)` | Stays valid when chunk size changes, so experiments are comparable. |
| Embedding cache keyed by `sha256(model, text)` | Re-ingesting after editing one lecture only embeds what changed. |
| Query traces (retrieved chunks + scores + timings + tokens) | Debug "wrong answer" by first checking whether retrieval found the right chunk. |
| Lambda + index in S3, not a vector DB | ~$0 at idle; a semester of notes fits in memory. See [infra/README.md](infra/README.md). |

## Quick start (local, $0)

```bash
uv venv && uv pip install -e ".[dev]"
make ingest
make ask Q="Why is ReLU preferred over sigmoid?"
make eval
```

| Want | Do |
|---|---|
| Real embeddings + reranker | `uv pip install -e ".[local]"` · `export STUDY_RAG_EMBEDDER=hf STUDY_RAG_RERANKER=hf` |
| Claude answers | `uv pip install -e ".[claude]"` · `export STUDY_RAG_LLM=claude` (API key) or `bedrock` (AWS creds) |
| Postgres + pgvector | `docker compose up -d` · `uv pip install -e ".[pg]"` · `export STUDY_RAG_STORE=pgvector` |
| HTTP API | `uv pip install -e ".[api]"` · `make serve` → `POST localhost:8080/ask` |
| One course only | `study-rag ask "..." --course machine-learning` |
| Grade answers | `make eval-answers` (costs tokens; capped by `--limit`) |
| Deploy to AWS | [infra/README.md](infra/README.md) |

## Using your own courses

Put files under `data/<course-name>/` (`.md`, `.txt`, `.pdf`). They're git-ignored,
because course materials are usually copyrighted. Add questions you actually had while
studying to `eval/golden.jsonl`.

## Build log

The core components are written by hand, without LangChain or LlamaIndex: the point is to
understand each piece. Guide: [docs/EXERCISES.md](docs/EXERCISES.md).

- [ ] 1 · Retrieval metrics (recall@k, MRR)
- [ ] 2 · Chunking
- [ ] 3 · Vector search + BM25
- [ ] 4 · Reciprocal Rank Fusion
- [ ] 5 · Grounded prompt + Claude call
- [ ] 6 · Cross-encoder reranking
- [ ] 7 · LLM-as-judge

## Project layout

```text
src/study_rag/
  ingest/      loaders (md/txt/pdf) · chunking
  store/       in-memory (numpy + BM25) · pgvector
  retrieval.py hybrid retriever · RRF · reranker
  llm.py       prompt assembly · Claude API / Bedrock / offline extractive
  eval/        retrieval metrics · LLM judge · answer eval
  cache.py     embedding cache          tracing.py  per-query JSONL traces
  api.py       FastAPI app              cli.py      study-rag CLI
prompts/       system prompts (versioned like code)
eval/          golden.jsonl · unanswerable.jsonl
infra/         Terraform: S3, ECR, Lambda, Function URL, OIDC, budget, alarms
specs/         spec-driven development: one spec per feature
```
