"""
Metric Service
==============
Fetches all monitored infrastructure metrics from Prometheus.
Returns results in a consistent structure for the detection loop.
"""

from ..prometheus_client import query_prometheus
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Metric definitions ────────────────────────────────────────────────────────
# Each entry defines a metric name and the PromQL query to retrieve it.
# Add new metrics here — the detection loop picks them up automatically.

METRIC_DEFINITIONS = [
    {
        "name":  "cpu_usage_percent",
        "query": (
            '100 - (avg by(instance) '
            '(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        ),
    },
    {
        "name":  "memory_usage_percent",
        "query": (
            '(1 - (node_memory_MemAvailable_bytes '
            '/ node_memory_MemTotal_bytes)) * 100'
        ),
    },
    {
        "name":  "disk_usage_percent",
        "query": (
            '(1 - (node_filesystem_avail_bytes{fstype!~"tmpfs|fuse.lxcfs"} '
            '/ node_filesystem_size_bytes)) * 100'
        ),
    },
    {
        "name":  "network_receive_bytes_per_sec",
        "query": 'rate(node_network_receive_bytes_total{device!="lo"}[5m])',
    },
]


def fetch_all_metrics() -> list[dict]:
    """
    Fetches all METRIC_DEFINITIONS from Prometheus.
    Returns a list of dicts with 'name' and 'results' (raw Prometheus result).
    Skips any metric that fails to query rather than raising.
    """
    collected = []
    for metric_def in METRIC_DEFINITIONS:
        try:
            response = query_prometheus(metric_def["query"])
            results  = response.get("data", {}).get("result", [])
            collected.append({
                "name":    metric_def["name"],
                "results": results,
            })
            logger.debug(
                "Fetched %d series for %s",
                len(results), metric_def["name"]
            )
        except Exception as exc:
            logger.error(
                "Failed to fetch metric %s: %s",
                metric_def["name"], exc
            )
    return collected


def fetch_cpu_metrics() -> dict:
    """
    Backwards-compatible single-metric fetch used by the /anomalies
    endpoint in the original detector stub.
    """
    try:
        return query_prometheus(METRIC_DEFINITIONS[0]["query"])
    except Exception as exc:
        logger.error("fetch_cpu_metrics failed: %s", exc)
        return {}
