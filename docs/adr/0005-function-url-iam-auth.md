# 0005. Function URL with IAM auth

**Status:** accepted

## Context
A public endpoint in front of a paid LLM is a bill waiting to happen. Single user.

## Decision
Lambda Function URL with `authorization_type = AWS_IAM` and response streaming. Callers
SigV4-sign requests (`awscurl`); unsigned requests never reach the code or the model.

## Alternatives rejected
- **API Gateway + API key:** more moving parts; REST APIs cannot stream.
- **Public URL with an app-level token:** the bill risk moves into application code.

## Consequences
- No user accounts: the AWS principal *is* the user. A multi-user version needs real
  authentication (Cognito + JWT authorizer) and per-user data, which is a different
  architecture (see ADR 0003) and out of scope for now.
- Browsers can't call it directly without SigV4 signing.
