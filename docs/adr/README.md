# Architecture decision records

One short file per decision that would be expensive to reverse. Format: context, decision,
alternatives rejected, consequences. A decision that changes gets a new ADR that supersedes
the old one; the old file stays, with its status updated.

| # | Decision | Status |
|---|---|---|
| [0001](0001-no-rag-framework.md) | No RAG framework: core on numpy and the Anthropic SDK | accepted |
| [0002](0002-hybrid-retrieval-rrf.md) | Hybrid retrieval fused with RRF | accepted |
| [0003](0003-index-in-s3-on-lambda.md) | Index file in S3 on Lambda, not a database | accepted |
| [0004](0004-safe-index-format.md) | Index without pickle, with build metadata | accepted |
| [0005](0005-function-url-iam-auth.md) | Function URL with IAM auth | accepted |
| [0006](0006-oidc-and-remote-state.md) | OIDC deploys and remote state | accepted |
| [0007](0007-emf-metrics-and-cost-alarm.md) | EMF metrics and an app-level cost alarm | accepted |
| [0008](0008-bedrock-mantle-iam.md) | Claude on Bedrock via the mantle endpoint | accepted, IAM unverified |
| [0009](0009-quiz-generation.md) | Quiz generation design | accepted (optional extension) |
| [0010](0010-baseline-regression-gate.md) | Retrieval regression gate against a baseline | accepted |
