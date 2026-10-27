# study-rag

A retrieval-augmented study assistant for my AI master's courses. Point it at lecture
notes, slides and papers, ask a question, and get an answer **grounded in the course
material, with citations**, instead of the model's general knowledge.

It is built like a service rather than a notebook demo: every retrieval change is scored
against a golden question set, every query leaves a trace with its cost, and the same code
runs locally for $0 or on AWS Lambda.

- **Measured.** Retrieval is scored with recall@k and MRR against a golden set on every PR;
  answers are graded for faithfulness by an LLM judge on a weekly schedule.
- **No RAG framework.** Retrieval, rank fusion, reranking and prompt assembly sit on numpy and
  the Anthropic SDK, so each decision is visible and can be measured
  ([ADR 0001](docs/adr/0001-no-rag-framework.md)).
- **Operated.** Per-query traces, cost per answer in USD, CloudWatch metrics and alarms, and a
  feedback loop that turns thumbs-down answers into new test cases.
- **Cheap to leave running.** Lambda + S3 with the index in S3, IAM-authenticated endpoint, OIDC
  deploys with no stored keys, reindexing when a document is uploaded. Idle cost is ~$0.

**Stack:** Python · Claude (Anthropic API or Amazon Bedrock) · sentence-transformers (BGE,
cross-encoder) · FastAPI · AWS Lambda, S3, CloudWatch · Terraform · GitHub Actions

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

Retrieval, scored against `eval/golden.jsonl`. Only measured numbers are reported here.

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

Longer write-ups, including what was rejected, are in [docs/adr/](docs/adr/).

| Decision | Why |
|---|---|
| Chunking: split on Markdown headings, then by size (800 characters, 120 overlap, defaults not yet tuned) | A chunk stays about one topic and carries its heading path (`Week 1 > Bias`) for citations. Sizes are the next thing to measure. |
| Hybrid search + RRF | Embeddings catch paraphrases, BM25 catches exact terms (acronyms, formula names). RRF fuses *ranks*, so no score calibration. |
| Cross-encoder rerank on top-N only | Precision where it matters at bounded latency. |
| Golden set labelled by `(source, phrase)` | Stays valid when chunk size changes, so experiments are comparable. |
| Embedding cache keyed by `sha256(model, text)` | Re-ingesting after editing one lecture only embeds what changed. |
| Query traces (retrieved chunks + scores + timings + tokens) | Debug "wrong answer" by first checking whether retrieval found the right chunk. |
| Index stores embedder/model/dim; refuses a mismatched config | Vectors from two models compare as plausible garbage. Fail loudly instead. |
| Index as `.npz` + JSON, never pickle | Loading a pickle executes code. |
| `prompt_version` (content hash) in every trace | A quality change can be tied to the exact prompt that caused it. |
| Cost per query from token usage, sent to CloudWatch (EMF) | "What does one answer cost?" has a number, with an alarm. |
| Thumbs-down feedback becomes golden-set candidates | The eval grows from real failures, not guesses. |
| Lambda + index in S3, reindex on upload | ~$0 at idle; course files never touch GitHub. See [infra/README.md](infra/README.md). |
| `uv.lock`, CPU-only torch pinned to the same version | Reproducible builds; the image doesn't ship 5 GB of CUDA. |

## Quick start (local, $0)

```bash
make install          # uv sync --frozen: exact versions from uv.lock
make ingest
make ask Q="Why is ReLU preferred over sigmoid?"
make eval
```

Out of the box this uses a deterministic hash embedder and extractive answers (no model, no
API key), which is what CI runs. Opt in to the real components:

| Want | Do |
|---|---|
| Real embeddings + reranker | `uv sync --extra local` · `export STUDY_RAG_EMBEDDER=hf STUDY_RAG_RERANKER=hf` |
| Claude answers | `uv sync --extra claude` · `export STUDY_RAG_LLM=claude` (API key) or `bedrock` (AWS creds) |
| Postgres + pgvector | `docker compose up -d` · `uv sync --extra pg` · `export STUDY_RAG_STORE=pgvector` |
| HTTP API | `uv sync --extra api` · `make serve` → `POST /ask`, `/ask/stream` (SSE), `/feedback` |
| One course only | `study-rag ask "..." --course machine-learning` |
| Quiz yourself (optional) | `make quiz` · `POST /quiz`; needs `STUDY_RAG_LLM=claude\|bedrock` |
| Fail a PR that makes retrieval worse | `make baseline` once, commit `eval/baseline.json` · `make eval` compares against it |
| Grade answers | `make eval-answers` (costs tokens; capped by `--limit`) |
| Turn thumbs-down answers into eval questions | `make candidates`, then label `eval/candidates.jsonl` |
| Deploy to AWS | [infra/README.md](infra/README.md) |

Configuration is environment variables with the `STUDY_RAG_` prefix, validated at startup
(see `src/study_rag/config.py`).

## Using your own courses

Put files under `data/<course-name>/` (`.md`, `.txt`, `.pdf`). They are git-ignored, because
course materials are usually copyrighted; only `data/sample/` is committed. Add questions you
actually had while studying to `eval/golden.jsonl`.

## Project layout

```text
src/study_rag/
  ingest/      loaders (md/txt/pdf) · chunking
  store/       in-memory (numpy + BM25) · pgvector
  retrieval.py hybrid retriever · RRF · reranker
  llm.py       prompt assembly · Claude API / Bedrock / offline extractive
  eval/        retrieval metrics · LLM judge · answer eval · baseline regression gate
  quiz/        chunk selection · structured-output generator · verifier (optional extension)
  cache.py     embedding cache          tracing.py  traces + CloudWatch EMF metrics
  pricing.py   token to USD             s3sync.py   S3 reindex + index watcher
  api.py       FastAPI app (+ SSE)      cli.py      study-rag CLI (Typer)
prompts/       system prompts (versioned like code)
eval/          golden.jsonl · unanswerable.jsonl
infra/         Terraform: bootstrap (state bucket) · S3, ECR, Lambda, Function URL,
               OIDC deploy role, S3 to Lambda trigger, dashboard, budget + cost/error alarms
.github/       ci (tests, eval gate, lockfile, terraform) · docker (build + smoke)
               deploy (manual, OIDC) · answer-eval (weekly LLM judge)
specs/         spec-driven development: one spec per feature (spec, plan, tasks)
docs/adr/      architecture decision records: why, and what was rejected
.specify/      project constitution (principles every change is checked against)
```

## Status and limits

- **Deployment:** the Terraform and the Lambda image are written and checked in CI, but not yet
  applied to a real AWS account, so cold-start time and the Bedrock IAM policy are unverified
  ([ADR 0008](docs/adr/0008-bedrock-mantle-iam.md)). Treat the cost figures as design targets
  until the dashboard has real traffic.
- **Scale:** a single user and an in-memory index, by design. The limits and what would change
  at scale are in [docs/adr/](docs/adr/).
- **pgvector** store has not been run against a real database.

Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md).
