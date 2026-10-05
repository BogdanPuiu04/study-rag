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
  description = "Where budget and error alerts go."
}

variable "github_repo" {
  type        = string
  description = "owner/name of the GitHub repo allowed to deploy."
}

variable "image_tag" {
  type        = string
  default     = ""
  description = "Leave empty on first apply; set after the first image push to create the Lambda."
}

variable "daily_llm_cost_alarm_usd" {
  type        = number
  default     = 1
  description = "Alert when Claude token spend in the last 24h exceeds this."
}

variable "claude_model" {
  type    = string
  default = "claude-opus-5-5"
}
