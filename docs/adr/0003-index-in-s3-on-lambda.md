# 0003. Index file in S3 on Lambda, not a database

**Status:** accepted

## Context
One student's course notes: thousands of chunks, not millions. Traffic is a few queries a
day. Requirement: ~$0 when idle (see `specs/001-core-rag`).

## Decision
The index is one `.npz` file in S3, loaded into memory by a Lambda container. Uploading to
`docs/` triggers a reindex; warm instances notice the new ETag (at most once a minute) and
reload. `pgvector` exists as a store for local use and for when the corpus outgrows memory.

## Alternatives rejected
- **RDS/Aurora + pgvector:** a provisioned instance bills 24/7; Aurora Serverless v2 can pause
  but adds a VPC or Data API layer and a resume delay.
- **OpenSearch Serverless:** hundreds of dollars a month at zero traffic.

## Consequences
- Idle cost is S3 and ECR cents.
- Single writer (reindex rebuilds the whole file); no per-user data.
- Switch to pgvector when the index no longer fits in Lambda memory or when several users
  need separate data.
