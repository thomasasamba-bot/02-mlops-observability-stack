"""
Anomaly Service
===============
Orchestrates the three detection algorithms (Z-Score, EWMA, Isolation Forest)
and produces a composite anomaly result with Prometheus metric updates.
"""

from datetime import datetime, timezone

from ..detection.zscore import detect_zscore
from ..detection.ewma import detect_ewma
from ..detection.isolation_forest import detect_isolation_forest
from ..metrics.exporter import (
    anomaly_score_gauge,
    zscore_gauge,
    ewma_gauge,
    isolation_forest_gauge,
    anomalies_total,
)
from ..config import ANOMALY_THRESHOLD
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Composite weights — must sum to 1.0
WEIGHTS = {
    "zscore":            0.40,
    "ewma":              0.30,
    "isolation_forest":  0.30,
}


def analyze_metric(metric_name: str, values: list[float]) -> list[dict]:
    """
    Runs all three detection algorithms on the provided value history.
    Updates Prometheus gauges for each method.
    Returns a list of anomaly dicts (empty if no anomaly detected).

    Args:
        metric_name: Human-readable metric name for labelling
        values:      Time-ordered list of float values (most recent last)

    Returns:
        List of anomaly result dicts. Empty list = no anomaly.
    """
    if len(values) < 5:
        return []

    current = values[-1]
    results = []

    # ── Run detection algorithms ──────────────────────────────────────
    try:
        z_anomaly, z_score = detect_zscore(values)
    except Exception as exc:
        logger.warning("Z-Score detection failed for %s: %s", metric_name, exc)
        z_anomaly, z_score = False, 0.0

    try:
        ewma_anomaly, ewma_score = detect_ewma(values)
    except Exception as exc:
        logger.warning("EWMA detection failed for %s: %s", metric_name, exc)
        ewma_anomaly, ewma_score = False, 0.0

    try:
        iso_anomaly, iso_score = detect_isolation_forest(values)
    except Exception as exc:
        logger.warning("Isolation Forest failed for %s: %s", metric_name, exc)
        iso_anomaly, iso_score = False, 0.0

    # ── Composite weighted score ──────────────────────────────────────
    composite_score = (
        z_score   * WEIGHTS["zscore"] +
        ewma_score * WEIGHTS["ewma"] +
        iso_score  * WEIGHTS["isolation_forest"]
    )

    # ── Update Prometheus gauges ──────────────────────────────────────
    zscore_gauge.labels(metric=metric_name).set(z_score)
    ewma_gauge.labels(metric=metric_name).set(ewma_score)
    isolation_forest_gauge.labels(metric=metric_name).set(iso_score)
    anomaly_score_gauge.labels(
        metric=metric_name, method="composite"
    ).set(composite_score)

    logger.debug(
        "%s | z=%.3f ew=%.3f if=%.3f composite=%.3f threshold=%.2f",
        metric_name, z_score, ewma_score, iso_score,
        composite_score, ANOMALY_THRESHOLD
    )

    # ── Emit result if anomaly detected ──────────────────────────────
    any_anomaly = z_anomaly or ewma_anomaly or iso_anomaly
    if any_anomaly or composite_score > ANOMALY_THRESHOLD:
        severity = "critical" if composite_score > 4.0 else "warning"
        anomalies_total.labels(metric=metric_name, severity=severity).inc()

        results.append({
            "metric_name":      metric_name,
            "anomaly_score":    round(composite_score, 4),
            "detection_method": _contributing_methods(z_anomaly, ewma_anomaly, iso_anomaly),
            "scores": {
                "zscore":           round(z_score, 4),
                "ewma":             round(ewma_score, 4),
                "isolation_forest": round(iso_score, 4),
                "composite":        round(composite_score, 4),
            },
            "value":     current,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    return results


def _contributing_methods(z: bool, ew: bool, iso: bool) -> str:
    """Returns a human-readable string of which methods flagged the anomaly."""
    methods = []
    if z:   methods.append("zscore")
    if ew:  methods.append("ewma")
    if iso: methods.append("isolation_forest")
    return "+".join(methods) if methods else "composite_threshold"
