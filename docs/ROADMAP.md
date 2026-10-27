# Roadmap

Where the project stands and what comes next. Updated whenever a spec's tasks change.

## State

| Area | State |
|---|---|
| Plumbing: CLI, API, tracing, cache, index format, S3 reindex | done, tested |
| Retrieval: chunking, vector search, BM25, RRF, cross-encoder rerank | done, tested; measured only on `data/sample` so far |
| Generation: grounded prompt, Claude call, streaming, prompt-injection hardening | done, tested with a fake client; live injection test not yet run |
| Answer eval: LLM judge | done, tested with a fake client; not yet run on real answers |
| Retrieval regression gate | done, waits for a baseline ([spec 003](../specs/003-regression-gate/spec.md)) |
| Quiz: writer prompt, verifier | done; not yet judged on real material ([spec 002](../specs/002-quiz/spec.md)) |
| Infra: Terraform, CI/CD, OIDC, dashboard, alarms | written, **never applied** ([spec 004](../specs/004-deploy-path/spec.md)) |
| Quality tooling: mypy, pre-commit, Dependabot, tflint | added; Trivy informational |

## Next, in order

**Milestone 1: real data, real numbers**
1. Real courses in `data/<course>/`; 30+ real study questions in `eval/golden.jsonl`.
2. `make eval` for the README results table (BM25 · hash · BGE · hybrid · rerank · chunk
   sizes 300/800/1500), with one sentence of interpretation under each. Then
   `make baseline` and commit `eval/baseline.json`.
3. Answer eval with the judge on a sample; run the prompt-injection test once.
4. CI fully green; the repo goes public.

**Milestone 2: deployed**
5. Deploy ([spec 004](../specs/004-deploy-path/spec.md) tasks): needs an AWS account, Bedrock
   model access, and a confirmed `terraform apply` per step.
6. Measure the Lambda cold start; dashboard screenshot and a short demo GIF in the README.

**Milestone 3: optional**
7. Judge noise (same eval twice) to set the regression threshold; harder PDFs (slides,
   formulas, scans); quiz quality on real material.
8. [Spec 005](../specs/005-embedding-finetune/spec.md) (embedding fine-tuning) only if the
   retrieval eval shows where retrieval fails.

## Considered and deliberately not done

| Idea | Why not |
|---|---|
| Aurora Serverless / pgvector on AWS | Spec 001 and ADR 0003: idle cost and added moving parts for a corpus that fits in memory. `pgvector` stays as a local store. |
| Cognito, multi-user accounts | Single-user by design (ADR 0005); multi-user changes the data architecture. |
| SageMaker endpoint for the Hugging Face models | The models are baked into the Lambda image; an endpoint adds a cold start and a bill to solve a problem not yet measured (measure the cold start first). |
| Agents / MCP | Out of scope before Milestone 1 (`GHID.md` §8: Milestone 3). |

## Unverified: read before trusting

- **Bedrock IAM** (`lambda.tf`): chosen from AWS documentation, not exercised. The first real
  `/ask` confirms it ([ADR 0008](adr/0008-bedrock-mantle-iam.md)).
- **Terraform and the Docker image** were not run locally (no Terraform installed in the
  session that wrote the changes): CI's `terraform fmt/validate/tflint` is the first check.
- **Trivy and tflint versions** in CI were pinned from memory of what existed; if a job fails on
  an action version rather than a finding, bump it.
- **`pgvector.py`** has never run against a real database.
