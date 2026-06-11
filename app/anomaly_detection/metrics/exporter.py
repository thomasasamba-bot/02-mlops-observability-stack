"""
Prometheus Metrics Exporter
============================
Defines all custom Prometheus metrics for the anomaly detection service.
Imported by anomaly_service.py to update gauges during detection cycles.
Exposed via /metrics endpoint (mounted in detector.py).
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Per-method anomaly scores ─────────────────────────────────────────────────

zscore_gauge = Gauge(
    "aiops_zscore_anomaly_score",
    "Z-Score anomaly score per metric",
    ["metric"],
)

ewma_gauge = Gauge(
    "aiops_ewma_anomaly_score",
    "EWMA anomaly score per metric",
    ["metric"],
)

isolation_forest_gauge = Gauge(
    "aiops_isolation_forest_anomaly_score",
    "Isolation Forest anomaly score per metric",
    ["metric"],
)

# ── Composite score (used in Grafana dashboards and alert rules) ──────────────

anomaly_score_gauge = Gauge(
    "aiops_infra_anomaly_score",
    "Composite infrastructure anomaly score per metric and detection method",
    ["metric", "method"],
)

# ── Outage probability ────────────────────────────────────────────────────────

outage_probability_gauge = Gauge(
    "aiops_outage_probability",
    "Predicted outage probability derived from composite anomaly scores (0-1)",
    ["instance"],
)

# ── Counters ──────────────────────────────────────────────────────────────────

anomalies_total = Counter(
    "aiops_anomalies_total",
    "Total number of anomalies detected since startup",
    ["metric", "severity"],
)

detection_cycles_total = Counter(
    "aiops_detection_cycles_total",
    "Total number of detection cycles completed",
)

# ── Detection cycle duration ──────────────────────────────────────────────────

detection_cycle_duration = Histogram(
    "aiops_detection_cycle_duration_seconds",
    "Time taken to complete one full detection cycle",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
