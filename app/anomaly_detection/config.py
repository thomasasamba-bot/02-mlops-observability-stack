"""
Configuration Module
====================

Centralized configuration for the AIOps Infrastructure
Anomaly Detection Service.

Configuration is loaded from environment variables
with sensible defaults for local Docker development.

Environment Variables
---------------------
PROMETHEUS_URL
    Prometheus API endpoint.

ALERTMANAGER_URL
    Alertmanager API endpoint.

ELASTICSEARCH_URL
    Elasticsearch endpoint.

ANOMALY_THRESHOLD
    Composite anomaly score threshold used to
    classify an observation as anomalous.

POLL_INTERVAL
    Number of seconds between Prometheus polling cycles.

LOG_LEVEL
    Application logging level.
"""

import os

# ============================================================================
# Service Endpoints
# ============================================================================

PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://prometheus:9090"
)

ALERTMANAGER_URL = os.getenv(
    "ALERTMANAGER_URL",
    "http://alertmanager:9093"
)

ELASTICSEARCH_URL = os.getenv(
    "ELASTICSEARCH_URL",
    "http://elasticsearch:9200"
)

# ============================================================================
# Detection Configuration
# ============================================================================

ANOMALY_THRESHOLD = float(
    os.getenv("ANOMALY_THRESHOLD", "2.5")
)

POLL_INTERVAL = int(
    os.getenv("POLL_INTERVAL", "30")
)

# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
)