# 0008. Claude on Bedrock via the mantle endpoint

**Status:** accepted. **The IAM statement has not been run against a real account.**

## Context
`BedrockLLM` uses `AnthropicBedrockMantle`, which calls the `bedrock-mantle` endpoint. That
endpoint is authorised by a different IAM service prefix than `bedrock-runtime`:

| Endpoint | IAM action |
|---|---|
| `bedrock-runtime` (InvokeModel / Converse) | `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` |
| `bedrock-mantle` (Messages API) | `bedrock-mantle:CreateInference` on a `project` resource |

The first version of `lambda.tf` granted the `bedrock:` actions, which would have failed with
`AccessDenied` on the first call, and carried a TODO to check.

## Decision
Grant `bedrock-mantle:CreateInference` on `arn:aws:bedrock-mantle:<region>:<account>:project/*`
and nothing else for Bedrock. Sources: AWS docs "Inference endpoints supported by Amazon
Bedrock" and the bedrock-mantle service authorization reference (actions, project ARN format).

## Alternatives
- **Switch to the bedrock-runtime endpoint** (also supports the Messages API, plus Guardrails
  and cross-region inference). A reasonable later move; it changes the client class and the
  IAM statement, and the model ID format.
- **`Resource: "*"`:** works, but grants every project in the account.

## Consequences
- If the first real call is denied, the error names the missing action (the docs mention
  read actions such as `GetProject` around the default project). Add it to the statement and
  update this record.
- Verify with `terraform plan` before the first `apply`, then one `/ask` through the Function URL.
