output "deploy_role_arn" {
  description = "Put this in the GitHub repo variable AWS_DEPLOY_ROLE_ARN."
  value       = aws_iam_role.deploy.arn
}

output "data_bucket" {
  value = aws_s3_bucket.data.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "function_url" {
  value = local.deploy_image ? aws_lambda_function_url.app[0].function_url : null
}
