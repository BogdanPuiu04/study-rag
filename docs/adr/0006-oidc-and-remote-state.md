# 0006. OIDC deploys and remote state

**Status:** accepted

## Context
CI must deploy without long-lived AWS keys in GitHub. Terraform state must survive a lost
laptop and refuse concurrent applies.

## Decision
GitHub Actions assumes an IAM role through OIDC. The trust policy matches
`repo:<owner>/<repo>:environment:production`: a job that names an environment gets that
subject *instead of* the branch, so the gate is the environment, and the environment's
"Deployment branches" setting (restrict to `main`) is what keeps it main-only. Deploys ship code only (`update-function-code`). State lives in an S3 bucket
(created by `infra/bootstrap`, versioned, `prevent_destroy`) with Terraform's native lockfile
(`use_lockfile`, Terraform >= 1.10), so there is no DynamoDB table to pay for or forget.

## Consequences
- An earlier version matched `ref:refs/heads/main` while `deploy.yml` ran in the `production`
  environment; those subjects never match, so the deploy could not have assumed the role.
- Nothing to rotate or leak in GitHub secrets.
- The deploy role can't read course data; data flows only through S3 uploads.
- Bootstrapping is two-phase (state bucket first), documented in `infra/README.md`.
