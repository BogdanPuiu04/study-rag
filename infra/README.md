# AWS deployment

```text
                     ┌──────────── code ─────────────┐        ┌──────────── data ────────────┐
 GitHub Actions ──OIDC──► deploy role ──► ECR image  │        │  make upload-docs            │
        └──────── update-function-code ──────┐       │        │   └► s3://…/docs/<course>/   │
                                             ▼       │        │          │ S3 event          │
                                   Lambda (FastAPI via Lambda Web Adapter) ◄─┘               │
   caller ──SigV4──► Function URL ──►  /ask, /ask/stream, /feedback   /events → reindex ──►  │
                     (AWS_IAM,           │                               s3://…/index/index.npz
                      streaming)         │ Claude on Bedrock              (+ embed cache)
                                         ▼
                CloudWatch Logs: query traces (JSON) ──EMF──► metrics: latency, tokens, cost
                     │                                          │
                     └── Logs Insights / dashboard              └── alarms → SNS email
```

## Why this shape

| Choice | Alternative | Why |
|---|---|---|
| Lambda + index file in S3 | ECS + RDS pgvector / OpenSearch | Pay per request; a database or vector cluster bills 24/7 even at zero traffic. A semester of notes fits in memory. |
| Reindex on S3 upload | Rebuild index in CI | Course files never touch GitHub (copyright). The embedding cache in S3 makes rebuilds cost only the new chunks. |
| Index watcher (ETag check ≤ 1/min) | Restart Lambda after reindex | Warm instances pick up the new index without a deploy. |
| `.npz` + JSON, no pickle | pickle | Loading a pickle executes code; a tampered index would be RCE. |
| Index records embedder + model + dim | Trust the config | Querying with another model returns garbage silently; now it fails loudly at startup. |
| EMF metrics in the log line | `PutMetricData` calls | Zero extra API calls/latency; metrics and the trace that produced them are the same record. |
| Cost alarm on token spend | Only AWS Budgets | Budgets lag up to a day; the metric alarm reacts within the hour. |
| Function URL, `AWS_IAM`, streaming | API Gateway + API key | Fewer moving parts; unsigned requests never reach your code (or Claude). API Gateway REST can't stream. |
| OIDC deploy role, no data access | Access keys in GitHub | Short-lived credentials, scoped to `main`; deploys ship code only. |
| Remote state in S3 + lockfile | Local state | Lose the laptop ≠ lose track of the infrastructure; no concurrent applies. |

When would you switch to pgvector on RDS? When the index no longer fits in Lambda
memory, or you need many writers / per-user data.

## Bootstrap (one time)

Prerequisites: AWS account, AWS CLI with admin credentials locally (`aws sso login`),
Terraform ≥ 1.10, Docker. In the Bedrock console, enable access to the Claude models
in your region. Before the first apply, resolve the TODO in `lambda.tf`.

```bash
# 0. State bucket (once per AWS account; keeps its own tiny state locally)
cd infra/bootstrap
terraform init && terraform apply
terraform output -raw backend_config > ../terraform/backend.hcl

# 1. Budget, buckets, ECR, roles, dashboard (no Lambda yet: no image exists)
cd ../terraform
cat > terraform.tfvars <<EOF
alert_email = "you@example.com"
github_repo = "<owner>/study-rag"
EOF
terraform init -backend-config=backend.hcl
terraform apply
# Confirm the subscription emails AWS sends you (budget + SNS alerts).

# 2. Push a first image by hand
REPO=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%/*}"
docker buildx build --provenance=false --platform linux/amd64 -t "$REPO:bootstrap" --push ../..

# 3. Create the Lambda + S3 trigger
terraform apply -var image_tag=bootstrap

# 4. Upload your course files: Lambda builds the index
cd ../.. && make upload-docs
```

Then in GitHub → Settings → Secrets and variables → Actions:
- **Variables**: `AWS_DEPLOY_ROLE_ARN` (output `deploy_role_arn`), `AWS_REGION`
- **Secrets** (optional): `ANTHROPIC_API_KEY` for the weekly answer-eval workflow
- **Environments**: create `production` (add yourself as required reviewer if you like)

From then on: code → **deploy** workflow; data → `make upload-docs`.

## Calling it

```bash
pip install awscurl
URL=$(cd infra/terraform && terraform output -raw function_url)
awscurl --service lambda -X POST "${URL}ask" \
  -H 'content-type: application/json' -d '{"question": "What is dropout?"}'
# streaming (Server-Sent Events):
awscurl --service lambda -X POST "${URL}ask/stream" \
  -H 'content-type: application/json' -d '{"question": "What is dropout?"}'
# rate an answer:
awscurl --service lambda -X POST "${URL}feedback" \
  -H 'content-type: application/json' -d '{"trace_id": "<from /ask>", "rating": -1}'
```

## Watching it

CloudWatch → Dashboards → **study-rag**: latency per stage, cost per hour, tokens,
Lambda errors, latest queries.

Export 👎 feedback + traces for `study-rag feedback-candidates` (Logs Insights on
`/aws/lambda/study-rag`, then download results as JSON lines):

```text
fields @message
| filter type = "feedback" or type = "query_trace"
| sort @timestamp asc
```

## Costs to expect (idle ≈ $0)

| Item | Cost |
|---|---|
| Lambda, S3, CloudWatch metrics from EMF | cents/month at personal use |
| ECR | ~$0.10/GB-month; lifecycle keeps 5 images (~1 GB each) |
| CloudWatch Logs | retention 14 days |
| Claude on Bedrock | per token; visible on the dashboard; alarm at `daily_llm_cost_alarm_usd` |

## Tearing down

`terraform destroy` in `infra/terraform` (empty the data bucket, all versions, and the
ECR repo first if it complains). The state bucket in `infra/bootstrap` has
`prevent_destroy`; remove it only after everything else is gone.
