# Roadmap

Where the project stands and what comes next, in the order that matters for the milestones
in `docs/private/GHID.md` §8. Updated whenever a spec's tasks change.

## State

| Area | State |
|---|---|
| Plumbing: CLI, API, tracing, cache, index format, S3 reindex | done, tested |
| Infra: Terraform, CI/CD, OIDC, dashboard, alarms | written, **never applied**; two defects fixed ([spec 004](../specs/004-deploy-path/spec.md)) |
| AI core: exercises 1–7, 5d | **not started** (the owner's work) |
| Retrieval regression gate | done, waits for a baseline ([spec 003](../specs/003-regression-gate/spec.md)) |
| Quiz | plumbing done; exercises 12a/12b open ([spec 002](../specs/002-quiz/spec.md)) |
| Quality tooling: mypy, pre-commit, Dependabot, tflint | added; Trivy informational |

## Next, in order

**Milestone 1: the repo can go public**
1. Setup (`GHID.md` §0), then exercises 1–5 one at a time, one commit each. Run
   `make test-plumbing` any time: it must stay green.
2. Real courses in `data/<course>/`; 30+ questions you actually had in `eval/golden.jsonl`.
3. `make eval` for the README results table (BM25 · hash · BGE · hybrid · chunk sizes), with
   one sentence of interpretation under each. Then `make baseline` and commit
   `eval/baseline.json`.
4. Fill the *Chunking* row in the README design table. CI fully green.

**Milestone 2: what sets it apart**
5. Exercises 6–8 (reranker, judge, prompt injection).
6. Deploy ([spec 004](../specs/004-deploy-path/spec.md) tasks): needs an AWS account, Bedrock
   model access, and your confirmation at each `terraform apply`.
7. Dashboard screenshot and a short demo GIF in the README; CV bullets from numbers you measured.

**Milestone 3: optional**
8. Exercises 9–11, then the quiz exercises 12a/12b.
9. [Spec 005](../specs/005-embedding-finetune/spec.md) (embedding fine-tuning on the Hugging
   Face stack) only if the retrieval eval shows where retrieval fails.

## Considered and deliberately not done

| Idea | Why not |
|---|---|
| Aurora Serverless / pgvector on AWS | Spec 001 and ADR 0003: idle cost and added moving parts for a corpus that fits in memory. `pgvector` stays as a local store. |
| Cognito, multi-user accounts | Single-user by design (ADR 0005); multi-user changes the data architecture. |
| SageMaker endpoint for the Hugging Face models | The models are baked into the Lambda image; an endpoint adds a cold start and a bill to solve a problem not yet measured (exercise 10 measures it first). |
| Agents / MCP | Out of scope before Milestone 1 (`GHID.md` §8: Milestone 3). |

## Unverified: read before trusting

- **Bedrock IAM** (`lambda.tf`): chosen from AWS documentation, not exercised. The first real
  `/ask` confirms it ([ADR 0008](adr/0008-bedrock-mantle-iam.md)).
- **Terraform and the Docker image** were not run locally (no Terraform installed in the
  session that wrote the changes): CI's `terraform fmt/validate/tflint` is the first check.
- **Trivy and tflint versions** in CI were pinned from memory of what existed; if a job fails on
  an action version rather than a finding, bump it.
- **`pgvector.py`** has never run against a real database.
