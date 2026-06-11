"""
Alertmanager Client
====================
Sends alert payloads to Alertmanager's /api/v2/alerts endpoint.
Called by the detection loop when an anomaly exceeds the threshold.
"""

import requests
from .config import ALERTMANAGER_URL
from .utils.logger import get_logger

logger = get_logger(__name__)


def send_alert(alert_payload: dict, timeout: int = 10) -> bool:
    """
    POST an alert payload to Alertmanager.

    Args:
        alert_payload: Single alert dict with 'labels', 'annotations',
                       and optionally 'startsAt' / 'endsAt'.
        timeout:       Request timeout in seconds.

    Returns:
        True if successfully sent, False otherwise.
    """
    url = f"{ALERTMANAGER_URL}/api/v2/alerts"
    try:
        response = requests.post(
            url,
            json=[alert_payload],   # Alertmanager expects a list
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        logger.info(
            "Alert sent to Alertmanager: %s",
            alert_payload.get("labels", {}).get("alertname", "unknown")
        )
        return True
    except requests.exceptions.ConnectionError:
        logger.warning("Alertmanager unreachable at %s — alert dropped", url)
        return False
    except requests.exceptions.Timeout:
        logger.warning("Alertmanager request timed out after %ds", timeout)
        return False
    except requests.exceptions.HTTPError as exc:
        logger.error("Alertmanager returned error: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error sending alert: %s", exc)
        return False


def resolve_alert(alert_name: str, instance: str) -> bool:
    """
    Resolves an active alert by sending it with an 'endsAt' timestamp.
    Alertmanager will mark it resolved.
    """
    from datetime import datetime, timezone
    payload = {
        "labels": {
            "alertname": alert_name,
            "instance":  instance,
        },
        "endsAt": datetime.now(timezone.utc).isoformat(),
    }
    return send_alert(payload)
