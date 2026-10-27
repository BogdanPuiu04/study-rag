# Tasks 004 — Deploy path

- [x] Bedrock-mantle IAM statement (docs-based)
- [x] OIDC trust matches the `production` environment; `infra/README.md` says to restrict
      the environment to `main`
- [x] tflint config + CI job; Trivy config scan (informational until triaged)
- [ ] **Owner, needs AWS:** Phase 0 setup (account, SSO, Bedrock model access, budget email)
- [ ] **Owner, needs AWS:** `terraform plan`/`apply` per `infra/README.md`, one decision at a time
- [ ] **Owner:** triage the first Trivy findings; add each exception to `.trivyignore` with a
      reason; make the job blocking
- [ ] Build the image once (`docker build .`) and read the `torch` install step's output
