# AWS deployment

```text
 GitHub Actions ──OIDC──► deploy role ──► ECR (image)  ─┐
        │                              └─► S3 (index)  ─┤
        │                                               ▼
        └──────── update-function-code ──────►  Lambda (FastAPI via Lambda Web Adapter)
                                                  │  loads index from S3 at cold start
   caller ──SigV4──► Function URL (AWS_IAM) ──────┘  calls Claude on Bedrock
                                                  │
                                       CloudWatch Logs (query traces) + Errors alarm → SNS email
```

## Why this shape

| Choice | Alternative | Why |
|---|---|---|
| Lambda + index file in S3 | ECS/EC2 + RDS pgvector or OpenSearch | Pay per request; a database or vector cluster bills 24/7 even at zero traffic. A semester of notes fits in memory. |
| Container image | Zip package | torch + models exceed the 250 MB zip limit; images go up to 10 GB. |
| Models baked into the image | Download at cold start | Faster, deterministic cold starts; no outbound traffic to Hugging Face. |
| Function URL, `AWS_IAM` auth | API Gateway + API key | Fewer moving parts; unsigned requests are rejected before your code (and Claude) runs. |
| OIDC deploy role | Access keys in GitHub secrets | Short-lived credentials, scoped to `main` of one repo. |
| Separate deploy and runtime roles | One role | Deploy can't call Claude; Lambda can't push images. Least privilege. |

When would you switch to pgvector on RDS? When the index no longer fits in Lambda
memory, or when you need incremental writes from multiple users.

## Bootstrap (one time)

Prerequisites: AWS account, AWS CLI with admin credentials locally (`aws sso login`),
Terraform ≥ 1.6, Docker. Bedrock: enable access to the Claude models in your region.

```bash
cd infra/terraform
cat > terraform.tfvars <<EOF
alert_email = "you@example.com"
github_repo = "<owner>/study-rag"
EOF

# Phase 1: budget, bucket, ECR, roles (no Lambda yet: no image exists)
terraform init && terraform apply
# Confirm the two subscription emails (budget + SNS) AWS sends you.

# Push a first image by hand
REPO=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%/*}"
docker build -t "$REPO:bootstrap" ../.. && docker push "$REPO:bootstrap"

# Upload an index (built locally — this is how your private course notes get there)
STUDY_RAG_EMBEDDER=hf study-rag ingest ../../data      # needs .[local]
aws s3 cp ../../.index/memory.pkl "s3://$(terraform output -raw data_bucket)/index/memory.pkl"

# Phase 2: create the Lambda
terraform apply -var image_tag=bootstrap
```

Then in GitHub → Settings → Secrets and variables → Actions → **Variables**:
`AWS_DEPLOY_ROLE_ARN` (output `deploy_role_arn`), `DATA_BUCKET` (output `data_bucket`),
`AWS_REGION`. Create an environment called `production`. From then on, the
**deploy** workflow does it all.

> The deploy workflow re-indexes `data/sample` only. Your real course notes stay
> off GitHub: build their index locally and upload it with the `aws s3 cp` above.

## Calling it

```bash
pip install awscurl
awscurl --service lambda -X POST "$(terraform output -raw function_url)ask" \
  -H 'content-type: application/json' -d '{"question": "What is dropout?"}'
```

## Watching it

CloudWatch Logs Insights on `/aws/lambda/study-rag`:

```text
fields @timestamp, question, timings_ms.retrieve, timings_ms.generate, usage.output_tokens
| filter type = "query_trace"
| sort @timestamp desc
```

## Tearing down

`terraform destroy`. Empty the S3 bucket (all versions) and ECR repo first if destroy complains.
