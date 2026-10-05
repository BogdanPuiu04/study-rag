variable "region" {
  type    = string
  default = "us-east-1"
}

variable "monthly_budget_usd" {
  type    = string
  default = "5"
}

variable "alert_email" {
  type        = string
  description = "Where budget alerts go."
}

variable "github_repo" {
  type        = string
  description = "owner/name of the GitHub repo allowed to assume the CI role."
}

output "ci_role_arn" {
  value = aws_iam_role.ci.arn
}

output "docs_bucket" {
  value = aws_s3_bucket.docs.bucket
}
