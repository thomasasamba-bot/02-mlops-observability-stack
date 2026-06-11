"""
Elasticsearch Client
=====================
Indexes anomaly events to Elasticsearch for historical analysis and Kibana dashboards.
Designed to degrade gracefully — if ES is unavailable, anomaly detection continues.
"""

from .config import ELASTICSEARCH_URL
from .utils.logger import get_logger

logger = get_logger(__name__)

# Lazy initialisation — only connect when first used
_es_client = None


def _get_client():
    """Returns a cached Elasticsearch client, initialising on first call."""
    global _es_client
    if _es_client is None:
        try:
            from elasticsearch import Elasticsearch
            _es_client = Elasticsearch(
                ELASTICSEARCH_URL,
                request_timeout=5,
                retry_on_timeout=True,
                max_retries=2,
            )
            logger.info("Elasticsearch client initialised: %s", ELASTICSEARCH_URL)
        except ImportError:
            logger.warning("elasticsearch package not installed — indexing disabled")
        except Exception as exc:
            logger.warning("Elasticsearch init failed: %s — indexing disabled", exc)
    return _es_client


def index_anomaly(document: dict) -> bool:
    """
    Indexes an anomaly document to the 'aiops-anomalies' index.

    Args:
        document: Dict containing anomaly data. Should include @timestamp.

    Returns:
        True if indexed successfully, False otherwise.
    """
    client = _get_client()
    if client is None:
        return False

    try:
        client.index(
            index="aiops-anomalies",
            document=document,
        )
        logger.debug(
            "Indexed anomaly: %s / %s score=%.3f",
            document.get("metric_name", "unknown"),
            document.get("instance", "unknown"),
            document.get("anomaly_score", 0),
        )
        return True
    except Exception as exc:
        logger.warning("Elasticsearch indexing failed: %s", exc)
        return False


def index_detection_cycle(cycle_summary: dict) -> bool:
    """Indexes a detection cycle summary for operational dashboards."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.index(index="aiops-detection-cycles", document=cycle_summary)
        return True
    except Exception as exc:
        logger.warning("Failed to index detection cycle: %s", exc)
        return False
