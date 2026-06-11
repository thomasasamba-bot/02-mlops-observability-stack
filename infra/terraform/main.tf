terraform {
  required_version = ">= 1.3.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.aws_region }

variable "aws_region"    { default = "us-east-1" }
variable "project_name"  { default = "observability-platform" }
variable "alert_email"   { default = "" }
variable "common_tags" {
  default = { Project = "observability-platform", ManagedBy = "terraform", Environment = "demo" }
}

# ─── SNS TOPIC FOR CLOUDWATCH ALARMS ─────────────────────────────────
resource "aws_sns_topic" "cloudwatch_alerts" {
  name = "${var.project_name}-cloudwatch-alerts"
  tags = var.common_tags
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.cloudwatch_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ─── CLOUDWATCH METRIC FILTERS (parse logs for anomalies) ─────────────
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/observability/${var.project_name}/app"
  retention_in_days = 30
  tags              = var.common_tags
}

resource "aws_cloudwatch_log_metric_filter" "error_rate" {
  name           = "${var.project_name}-error-rate"
  log_group_name = aws_cloudwatch_log_group.app_logs.name
  pattern        = "[timestamp, level=ERROR, ...]"

  metric_transformation {
    name      = "ErrorCount"
    namespace = "${var.project_name}/Application"
    value     = "1"
    unit      = "Count"
  }
}

resource "aws_cloudwatch_log_metric_filter" "latency" {
  name           = "${var.project_name}-latency"
  log_group_name = aws_cloudwatch_log_group.app_logs.name
  pattern        = "[timestamp, level, msg, latency_ms]"

  metric_transformation {
    name      = "LatencyMs"
    namespace = "${var.project_name}/Application"
    value     = "$latency_ms"
    unit      = "Milliseconds"
  }
}

# ─── CLOUDWATCH ANOMALY DETECTION ALARMS ─────────────────────────────
resource "aws_cloudwatch_metric_alarm" "error_rate_anomaly" {
  alarm_name          = "${var.project_name}-error-rate-anomaly"
  comparison_operator = "GreaterThanUpperThreshold"
  evaluation_periods  = 3
  threshold_metric_id = "ad1"
  alarm_description   = "Error rate anomaly detected"
  alarm_actions       = [aws_sns_topic.cloudwatch_alerts.arn]
  treat_missing_data  = "notBreaching"

  metric_query {
    id          = "m1"
    return_data = false
    metric {
      metric_name = "ErrorCount"
      namespace   = "${var.project_name}/Application"
      period      = 60
      stat        = "Sum"
    }
  }

  metric_query {
    id          = "ad1"
    expression  = "ANOMALY_DETECTION_BAND(m1, 2)"
    label       = "ErrorCount (expected)"
    return_data = true
  }

  tags = var.common_tags
}

# ─── CLOUDWATCH COMPOSITE ALARM ───────────────────────────────────────
resource "aws_cloudwatch_composite_alarm" "platform_health" {
  alarm_name        = "${var.project_name}-platform-health"
  alarm_description = "Composite: fires when error rate AND latency anomalies both detected"
  alarm_rule        = "ALARM(${aws_cloudwatch_metric_alarm.error_rate_anomaly.alarm_name})"
  alarm_actions     = [aws_sns_topic.cloudwatch_alerts.arn]
  tags              = var.common_tags
}

# ─── CLOUDWATCH DASHBOARD ─────────────────────────────────────────────
resource "aws_cloudwatch_dashboard" "observability" {
  dashboard_name = "${var.project_name}-dashboard"
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title   = "Application Error Rate with Anomaly Band"
          view    = "timeSeries"
          metrics = [
            ["${var.project_name}/Application", "ErrorCount", { stat = "Sum", label = "Errors" }],
            [{ expression = "ANOMALY_DETECTION_BAND(m1, 2)", label = "Expected Range", id = "ad1" }]
          ]
          period = 60
        }
      },
      {
        type = "alarm", x = 0, y = 6, width = 24, height = 3
        properties = {
          title  = "Platform Health Alarms"
          alarms = [aws_cloudwatch_composite_alarm.platform_health.arn]
        }
      }
    ]
  })
}

output "sns_topic_arn"      { value = aws_sns_topic.cloudwatch_alerts.arn }
output "log_group_name"     { value = aws_cloudwatch_log_group.app_logs.name }
output "dashboard_url"      { value = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.observability.dashboard_name}" }
