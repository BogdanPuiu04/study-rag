# --- Lambda runtime role: read the index, call Claude, write logs. Nothing else.
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
    sid       = "ReadIndex"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/${local.index_key}"]
  }
  statement {
    sid = "InvokeClaude"
    # TODO: confirm the exact IAM actions for the Bedrock Messages-API (Mantle)
    # endpoint in the Bedrock docs before first apply.
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = ["*"]
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

# --- Logs + alarm ----------------------------------------------------------------
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

# --- The function (created once an image has been pushed; see infra/README.md) ----
resource "aws_lambda_function" "app" {
  count         = local.deploy_image ? 1 : 0
  function_name = local.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  architectures = ["x86_64"]
  memory_size   = 2048 # torch + two small models; also buys CPU (Lambda scales CPU with memory)
  timeout       = 60

  environment {
    variables = {
      STUDY_RAG_INDEX_S3     = "s3://${aws_s3_bucket.data.bucket}/${local.index_key}"
      STUDY_RAG_LLM          = "bedrock"
      STUDY_RAG_CLAUDE_MODEL = var.claude_model
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
}

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
