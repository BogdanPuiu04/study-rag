# Spec 004 — The deploy path works end to end

**Status:** two defects fixed in code; **nothing verified against a real AWS account.**

## Problem
Nothing has been deployed, so the path from `git push` to an answered `/ask` has never run.
Reading it end to end found two defects that would have failed the first deploy.

## Defects found and fixed
1. **Bedrock IAM.** The Lambda role granted `bedrock:InvokeModel*`, but `BedrockLLM` calls the
   bedrock-mantle endpoint, authorised by `bedrock-mantle:CreateInference` on a project
   resource. Fixed in `infra/terraform/lambda.tf` ([ADR 0008](../../docs/adr/0008-bedrock-mantle-iam.md)).
2. **OIDC trust.** The deploy role trusted `ref:refs/heads/main`, but `deploy.yml` runs in the
   `production` environment, where GitHub sends `environment:production` as `sub` instead.
   Fixed in `infra/terraform/github_oidc.tf` ([ADR 0006](../../docs/adr/0006-oidc-and-remote-state.md)).

## Acceptance criteria (all open: they need an AWS account)
- [ ] `terraform plan` on `infra/bootstrap`, then `infra/terraform`, shows only expected resources.
- [ ] Deploy workflow assumes the role and rolls the Lambda.
- [ ] `/health` through the Function URL returns the index metadata.
- [ ] `/ask` returns a cited answer: confirms (or corrects) the Bedrock IAM statement.
- [ ] Cold start measured (exercise 10); Docker image built once (`docker.yml` has never run).

## Out of scope
Multi-user auth, a database, a UI.
