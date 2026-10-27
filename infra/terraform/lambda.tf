# --- Lambda runtime role: read docs, read/write index + cache, call Claude, write logs.
data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "lambda" {
  statement {
    sid       = "ListDocs"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${local.docs_prefix}*"]
    }
  }
  statement {
    sid       = "ReadDocs"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/${local.docs_prefix}*"]
  }
  statement {
    sid     = "ReadWriteIndex"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/${local.index_key}",
      "${aws_s3_bucket.data.arn}/${local.cache_key}",
    ]
  }
  statement {
    sid = "InvokeClaude"

    # BedrockLLM uses AnthropicBedrockMantle (bedrock-mantle endpoint). Its IAM action is
    # bedrock-mantle:CreateInference, on a *project* resource: not bedrock:InvokeModel,
    # which belongs to the bedrock-runtime endpoint. Source: "Inference endpoints supported
    # by Amazon Bedrock" and the bedrock-mantle service authorization reference.
    # Not exercised against a real account yet: if the first call is AccessDenied, the
    # error names the missing action; add it here (see docs/adr/0008-bedrock-mantle-iam.md).
    actions   = ["bedrock-mantle:CreateInference"]
    resources = ["arn:aws:bedrock-mantle:${var.region}:${data.aws_caller_identity.current.account_id}:project/*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.lambda.arn}:*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

# --- Logs + alerts -----------------------------------------------------------------
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = 14 # logs cost money forever by default
}

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# --- The function (created once an image has been pushed; see infra/README.md) ------
resource "aws_lambda_function" "app" {
  count         = local.deploy_image ? 1 : 0
  function_name = local.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  architectures = ["x86_64"]
  memory_size   = 2048 # torch + two small models; also buys CPU (Lambda scales CPU with memory)
  timeout       = 300  # reindexing a semester of PDFs on CPU; queries take seconds

  ephemeral_storage {
    # /tmp holds the downloaded docs, index and embedding cache during reindex.
    # The 512 MB default fills up with a semester of slide PDFs.
    size = 2048
  }

  environment {
    variables = {
      STUDY_RAG_INDEX_S3     = "s3://${aws_s3_bucket.data.bucket}/${local.index_key}"
      STUDY_RAG_DOCS_S3      = "s3://${aws_s3_bucket.data.bucket}/${local.docs_prefix}"
      STUDY_RAG_CACHE_S3     = "s3://${aws_s3_bucket.data.bucket}/${local.cache_key}"
      STUDY_RAG_LLM          = "bedrock"
      STUDY_RAG_CLAUDE_MODEL = var.claude_model
      AWS_LWA_INVOKE_MODE    = "response_stream" # lets /ask/stream stream through the URL
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda, aws_iam_role_policy.lambda]

  lifecycle {
    # CI deploys new images with `aws lambda update-function-code`.
    ignore_changes = [image_uri]
  }
}

# IAM auth: callers must SigV4-sign requests, so a leaked URL can't run up a Claude bill.
resource "aws_lambda_function_url" "app" {
  count              = local.deploy_image ? 1 : 0
  function_name      = aws_lambda_function.app[0].function_name
  authorization_type = "AWS_IAM"
  invoke_mode        = "RESPONSE_STREAM"
}

# --- Upload under docs/ -> reindex ------------------------------------------------------
resource "aws_lambda_permission" "s3_invoke" {
  count          = local.deploy_image ? 1 : 0
  statement_id   = "AllowS3Invoke"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.app[0].function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.data.arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket_notification" "docs" {
  count  = local.deploy_image ? 1 : 0
  bucket = aws_s3_bucket.data.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.app[0].arn
    events              = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
    # Only docs/: the index is written to index/, so reindexing can't trigger itself.
    filter_prefix = local.docs_prefix
  }
  depends_on = [aws_lambda_permission.s3_invoke]
}

# --- Alarms --------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "errors" {
  count               = local.deploy_image ? 1 : 0
  alarm_name          = "${local.name}-errors"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.app[0].function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "daily_llm_cost" {
  # App-level cost from token usage (StudyRAG/CostUSD, emitted via EMF): reacts within
  # an hour, unlike AWS Budgets which lags by up to a day.
  alarm_name          = "${local.name}-llm-cost-24h"
  evaluation_periods  = 1
  threshold           = var.daily_llm_cost_alarm_usd
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "cost"
    expression  = "SUM(SEARCH('{StudyRAG,Model} MetricName=\"CostUSD\"', 'Sum', 86400))"
    label       = "LLM cost (USD, 24h)"
    return_data = true
    period      = 86400
  }
}
