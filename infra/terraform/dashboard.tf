# One screen: is it up, is it fast, what does it cost, is retrieval/answering healthy.
# Metrics come from the EMF block in each query trace (src/study_rag/tracing.py).

locals {
  search = "{StudyRAG,Model} MetricName"
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = local.name
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title  = "Latency per stage (p50 / p95, ms)"
          region = var.region
          view   = "timeSeries"
          metrics = [
            [{ expression = "SEARCH('${local.search}=\"RetrieveMs\"', 'p50', 300)", label = "retrieve p50", id = "r50" }],
            [{ expression = "SEARCH('${local.search}=\"RetrieveMs\"', 'p95', 300)", label = "retrieve p95", id = "r95" }],
            [{ expression = "SEARCH('${local.search}=\"GenerateMs\"', 'p50', 300)", label = "generate p50", id = "g50" }],
            [{ expression = "SEARCH('${local.search}=\"GenerateMs\"', 'p95', 300)", label = "generate p95", id = "g95" }],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "LLM cost (USD per hour)"
          region = var.region
          view   = "timeSeries"
          stat   = "Sum"
          metrics = [
            [{ expression = "SEARCH('${local.search}=\"CostUSD\"', 'Sum', 3600)", label = "cost", id = "c" }],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title  = "Tokens per hour"
          region = var.region
          view   = "timeSeries"
          metrics = [
            [{ expression = "SEARCH('${local.search}=\"InputTokens\"', 'Sum', 3600)", label = "input", id = "ti" }],
            [{ expression = "SEARCH('${local.search}=\"OutputTokens\"', 'Sum', 3600)", label = "output", id = "to" }],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title  = "Lambda: invocations, errors, cold starts (init duration)"
          region = var.region
          view   = "timeSeries"
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", local.name, { stat = "Sum" }],
            ["AWS/Lambda", "Errors", "FunctionName", local.name, { stat = "Sum" }],
            ["AWS/Lambda", "Duration", "FunctionName", local.name, { stat = "p95" }],
          ]
        }
      },
      {
        type = "log", x = 0, y = 12, width = 24, height = 6
        properties = {
          title  = "Latest queries"
          region = var.region
          query  = <<-EOQ
            SOURCE '/aws/lambda/${local.name}'
            | fields @timestamp, question, prompt_version, timings_ms.generate, cost_usd, error
            | filter type = "query_trace"
            | sort @timestamp desc
            | limit 20
          EOQ
        }
      },
    ]
  })
}
