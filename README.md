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
| Index stores embedder/model/dim; refuses a mismatched config | Vectors from two models compare as plausible garbage. Fail loudly instead. |
| Index as `.npz` + JSON, never pickle | Loading a pickle executes code. |
| `prompt_version` (content hash) in every trace | A quality change can be tied to the exact prompt that caused it. |
| Cost per query from token usage → CloudWatch (EMF) | "What does one answer cost?" has a number, with an alarm. |
| 👎 feedback → golden-set candidates | The eval grows from real failures, not guesses. |
| Lambda + index in S3, reindex on upload | ~$0 at idle; course files never touch GitHub. See [infra/README.md](infra/README.md). |
| `uv.lock`, CPU-only torch pinned to the same version | Reproducible builds; the image doesn't ship 5 GB of CUDA. |

## Quick start (local, $0)

```bash
make install          # uv sync --frozen: exact versions from uv.lock
make ingest
make ask Q="Why is ReLU preferred over sigmoid?"
make eval
```

| Want | Do |
|---|---|
| Real embeddings + reranker | `uv sync --extra local` · `export STUDY_RAG_EMBEDDER=hf STUDY_RAG_RERANKER=hf` |
| Claude answers | `uv sync --extra claude` · `export STUDY_RAG_LLM=claude` (API key) or `bedrock` (AWS creds) |
| Postgres + pgvector | `docker compose up -d` · `uv sync --extra pg` · `export STUDY_RAG_STORE=pgvector` |
| HTTP API | `uv sync --extra api` · `make serve` → `POST /ask`, `/ask/stream` (SSE), `/feedback` |
| One course only | `study-rag ask "..." --course machine-learning` |
| Grade answers | `make eval-answers` (costs tokens; capped by `--limit`) |
| Turn 👎 into eval questions | `make candidates` → label `eval/candidates.jsonl` |
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
- [ ] 5d · Streaming generation
- [ ] 8 · Prompt-injection defence (attack test first)
- [ ] 9 · Quality regressions over time
- [ ] 10 · Cold start: measured, then decided
- [ ] 11 · Hard PDFs: slides, formulas, scans

## Project layout

```text
src/study_rag/
  ingest/      loaders (md/txt/pdf) · chunking
  store/       in-memory (numpy + BM25) · pgvector
  retrieval.py hybrid retriever · RRF · reranker
  llm.py       prompt assembly · Claude API / Bedrock / offline extractive
  eval/        retrieval metrics · LLM judge · answer eval
  cache.py     embedding cache          tracing.py  traces + CloudWatch EMF metrics
  pricing.py   token → USD              s3sync.py   S3 reindex + index watcher
  api.py       FastAPI app (+ SSE)      cli.py      study-rag CLI
prompts/       system prompts (versioned like code)
eval/          golden.jsonl · unanswerable.jsonl
infra/         Terraform: bootstrap (state bucket) · S3, ECR, Lambda, Function URL,
               OIDC deploy role, S3→Lambda trigger, dashboard, budget + cost/error alarms
.github/       ci (tests, eval gate, lockfile, terraform) · docker (build + smoke)
               deploy (manual, OIDC) · answer-eval (weekly LLM judge)
specs/         spec-driven development: one spec per feature
```
