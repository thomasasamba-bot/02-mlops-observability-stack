"""
app/pipeline/predict.py

ML inference logic for the credit scoring model.

Responsibilities:
  1. Load the trained model from MLflow Model Registry (or a local run URI)
  2. Validate incoming feature payloads against the training schema
  3. Run predictions and return probability scores + binary decisions
  4. Track prediction confidence distribution as Prometheus metrics
  5. Log prediction records to a rolling in-memory buffer (consumed by
     drift_detector.py and the FastAPI serving layer in app/serving/app.py)

Design decisions:
  - Model is loaded once at startup and cached (lazy singleton pattern)
  - Reload is triggered when a newer "Production" or "Staging" version exists
  - Low-confidence predictions (proba between LOW_CONF_LOW and LOW_CONF_HIGH)
    are flagged so the serving layer can route them for human review
  - All public functions are safe to call from multiple threads (the serving
    app runs with multiple Uvicorn workers)

Usage (standalone smoke test):
  python -m app.pipeline.predict
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "num_credit_lines",
    "missed_payments",
]

MODEL_NAME        = "credit-scoring-model"
MODEL_STAGE       = os.getenv("MODEL_STAGE", "Staging")   # promote to Production after validation
MODEL_ALIAS       = os.getenv("MODEL_ALIAS", None)        # optional: @champion alias
SCHEMA_PATH       = Path(os.getenv("SCHEMA_PATH", "data/processed/feature_schema.json"))
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Confidence thresholds
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.40"))  # lower than 0.5 → recall-focused
LOW_CONF_LOW       = float(os.getenv("LOW_CONF_LOW",  "0.35"))
LOW_CONF_HIGH      = float(os.getenv("LOW_CONF_HIGH", "0.55"))

# Rolling buffer — last N predictions kept for drift_detector.py
PREDICTION_BUFFER_SIZE = int(os.getenv("PREDICTION_BUFFER_SIZE", "2000"))

# Model reload check interval (seconds)
MODEL_RELOAD_INTERVAL = int(os.getenv("MODEL_RELOAD_INTERVAL", "300"))


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PRED_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Prediction probability scores (positive class)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PRED_TOTAL = Counter(
    "ml_predictions_total",
    "Total predictions served",
    ["decision"],  # labels: "default" | "no_default"
)

PRED_LOW_CONFIDENCE = Counter(
    "ml_predictions_low_confidence_total",
    "Predictions falling in the uncertain confidence band",
)

PRED_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "End-to-end prediction latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

MODEL_VERSION_GAUGE = Gauge(
    "ml_model_version",
    "Currently loaded MLflow model version",
)

MODEL_LOAD_ERRORS = Counter(
    "ml_model_load_errors_total",
    "Number of times model loading failed",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """Single prediction result — stored in the rolling buffer."""
    timestamp:   float
    features:    dict[str, float]
    probability: float
    decision:    int          # 0 or 1
    low_conf:    bool
    model_version: str


@dataclass
class PredictionResult:
    """Returned to callers of predict() / predict_batch()."""
    probability:   float
    decision:      int
    low_confidence: bool
    model_version: str
    latency_ms:    float
    features_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model cache (singleton with thread-safe reload)
# ---------------------------------------------------------------------------

class _ModelCache:
    """
    Thread-safe singleton that holds the loaded sklearn Pipeline.
    Periodically checks MLflow for a newer version and hot-reloads.
    """

    def __init__(self) -> None:
        self._lock          = threading.RLock()
        self._model         = None
        self._model_version = "unknown"
        self._last_check    = 0.0
        self._schema: dict  = {}

    # ── Schema ────────────────────────────────────────────────────────────

    def _load_schema(self) -> None:
        if SCHEMA_PATH.exists():
            self._schema = json.loads(SCHEMA_PATH.read_text())
            log.info("Feature schema loaded from %s", SCHEMA_PATH)
        else:
            log.warning(
                "Feature schema not found at %s — validation disabled. "
                "Run app/pipeline/train.py to generate it.",
                SCHEMA_PATH,
            )

    # ── Model loading ─────────────────────────────────────────────────────

    def _resolve_model_uri(self) -> tuple[str, str]:
        """
        Returns (model_uri, version_label).
        Tries MLflow registry first; falls back to local mlruns artefact.
        """
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        try:
            if MODEL_ALIAS:
                mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
                uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
                return uri, mv.version
        except Exception:
            pass  # alias not set, fall through

        try:
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if versions:
                mv = versions[0]
                uri = f"models:/{MODEL_NAME}/{mv.version}"
                return uri, mv.version
        except Exception as exc:
            log.warning("MLflow registry lookup failed: %s", exc)

        raise RuntimeError(
            f"No model found in MLflow registry (name={MODEL_NAME}, stage={MODEL_STAGE}). "
            "Run app/pipeline/train.py first."
        )

    def _do_load(self) -> None:
        try:
            uri, version = self._resolve_model_uri()
            log.info("Loading model from %s …", uri)
            model = mlflow.sklearn.load_model(uri)
            with self._lock:
                self._model         = model
                self._model_version = str(version)
                MODEL_VERSION_GAUGE.set(float(version) if version.isdigit() else 0)
            log.info("Model loaded  version=%s", version)
        except Exception as exc:
            MODEL_LOAD_ERRORS.inc()
            log.error("Failed to load model: %s", exc)
            raise

    # ── Public interface ──────────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        now = time.monotonic()
        with self._lock:
            needs_check = (now - self._last_check) > MODEL_RELOAD_INTERVAL
            is_loaded   = self._model is not None

        if not is_loaded:
            self._load_schema()
            self._do_load()
            with self._lock:
                self._last_check = time.monotonic()
        elif needs_check:
            # Background reload — don't block the request
            t = threading.Thread(target=self._background_reload, daemon=True)
            t.start()
            with self._lock:
                self._last_check = time.monotonic()

    def _background_reload(self) -> None:
        try:
            self._do_load()
        except Exception:
            pass  # already logged in _do_load; keep serving with existing model

    @property
    def model(self):
        with self._lock:
            return self._model

    @property
    def version(self) -> str:
        with self._lock:
            return self._model_version

    @property
    def schema(self) -> dict:
        return self._schema


_cache = _ModelCache()

# Rolling prediction buffer (thread-safe deque)
_prediction_buffer: deque[PredictionRecord] = deque(maxlen=PREDICTION_BUFFER_SIZE)
_buffer_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    pass


def validate_features(data: dict[str, Any]) -> pd.DataFrame:
    """
    Validate and coerce a feature dict into a single-row DataFrame.
    Raises ValidationError with a descriptive message on failure.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        raise ValidationError(f"Missing required features: {missing}")

    extra = [k for k in data if k not in FEATURE_COLUMNS]
    if extra:
        log.debug("Ignoring extra keys in payload: %s", extra)

    row: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        try:
            row[col] = float(data[col])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Feature '{col}' must be numeric: {exc}") from exc

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # Range validation from schema (warn only — don't reject live traffic)
    schema_features = _cache.schema.get("features", {})
    for col, stats in schema_features.items():
        val = row[col]
        lo, hi = stats["min"] * 0.5, stats["max"] * 1.5  # generous bounds
        if not (lo <= val <= hi):
            log.warning(
                "Feature '%s' value %.2f is far outside training range [%.2f, %.2f]",
                col, val, stats["min"], stats["max"],
            )

    return df


def validate_batch(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate a list of feature dicts → DataFrame with one row per record."""
    if not records:
        raise ValidationError("Empty batch")
    frames = [validate_features(r) for r in records]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _make_prediction(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Core inference — returns (probabilities, decisions)."""
    _cache.ensure_loaded()
    model = _cache.model
    if model is None:
        raise RuntimeError("Model not loaded")

    probas    = model.predict_proba(df)[:, 1]
    decisions = (probas >= DECISION_THRESHOLD).astype(int)
    return probas, decisions


def predict(features: dict[str, Any]) -> PredictionResult:
    """
    Single-record prediction.

    Args:
        features: dict mapping feature name → value

    Returns:
        PredictionResult with probability, decision, confidence flag, latency
    """
    t0  = time.perf_counter()
    df  = validate_features(features)
    probas, decisions = _make_prediction(df)
    latency = (time.perf_counter() - t0) * 1000  # ms

    prob     = float(probas[0])
    decision = int(decisions[0])
    low_conf = LOW_CONF_LOW <= prob <= LOW_CONF_HIGH

    # Prometheus
    PRED_CONFIDENCE.observe(prob)
    PRED_TOTAL.labels(decision="default" if decision == 1 else "no_default").inc()
    PRED_LATENCY.observe(latency / 1000)
    if low_conf:
        PRED_LOW_CONFIDENCE.inc()

    # Buffer
    record = PredictionRecord(
        timestamp=time.time(),
        features=features,
        probability=prob,
        decision=decision,
        low_conf=low_conf,
        model_version=_cache.version,
    )
    with _buffer_lock:
        _prediction_buffer.append(record)

    log.debug(
        "predict  prob=%.4f  decision=%d  low_conf=%s  latency=%.2fms  version=%s",
        prob, decision, low_conf, latency, _cache.version,
    )

    return PredictionResult(
        probability=prob,
        decision=decision,
        low_confidence=low_conf,
        model_version=_cache.version,
        latency_ms=round(latency, 3),
        features_used=FEATURE_COLUMNS,
    )


def predict_batch(records: list[dict[str, Any]]) -> list[PredictionResult]:
    """
    Batch prediction — more efficient than calling predict() in a loop.

    Args:
        records: list of feature dicts

    Returns:
        list of PredictionResult (same order as input)
    """
    t0  = time.perf_counter()
    df  = validate_batch(records)
    probas, decisions = _make_prediction(df)
    latency_total = (time.perf_counter() - t0) * 1000
    latency_per   = latency_total / len(records)

    results = []
    for i, (prob, decision) in enumerate(zip(probas.tolist(), decisions.tolist())):
        prob     = float(prob)
        decision = int(decision)
        low_conf = LOW_CONF_LOW <= prob <= LOW_CONF_HIGH

        PRED_CONFIDENCE.observe(prob)
        PRED_TOTAL.labels(decision="default" if decision == 1 else "no_default").inc()
        PRED_LATENCY.observe(latency_per / 1000)
        if low_conf:
            PRED_LOW_CONFIDENCE.inc()

        record = PredictionRecord(
            timestamp=time.time(),
            features=records[i],
            probability=prob,
            decision=decision,
            low_conf=low_conf,
            model_version=_cache.version,
        )
        with _buffer_lock:
            _prediction_buffer.append(record)

        results.append(PredictionResult(
            probability=prob,
            decision=decision,
            low_confidence=low_conf,
            model_version=_cache.version,
            latency_ms=round(latency_per, 3),
            features_used=FEATURE_COLUMNS,
        ))

    log.info(
        "predict_batch  n=%d  total_latency=%.2fms  per_record=%.2fms",
        len(records), latency_total, latency_per,
    )
    return results


# ---------------------------------------------------------------------------
# Buffer access (for drift_detector.py)
# ---------------------------------------------------------------------------

def get_recent_predictions(n: int | None = None) -> list[PredictionRecord]:
    """
    Return the last n prediction records from the rolling buffer.
    If n is None, return all buffered records.
    Used by drift_detector.py to sample recent inference traffic.
    """
    with _buffer_lock:
        records = list(_prediction_buffer)
    return records[-n:] if n and n < len(records) else records


def get_prediction_stats() -> dict[str, Any]:
    """Summary stats for the /status endpoint in app/serving/app.py."""
    records = get_recent_predictions()
    if not records:
        return {"buffered_predictions": 0}

    probs = [r.probability for r in records]
    return {
        "buffered_predictions":  len(records),
        "model_version":         _cache.version,
        "decision_threshold":    DECISION_THRESHOLD,
        "confidence": {
            "mean":  round(float(np.mean(probs)), 4),
            "std":   round(float(np.std(probs)), 4),
            "p25":   round(float(np.percentile(probs, 25)), 4),
            "p50":   round(float(np.percentile(probs, 50)), 4),
            "p75":   round(float(np.percentile(probs, 75)), 4),
        },
        "default_rate": round(
            sum(1 for r in records if r.decision == 1) / len(records), 4
        ),
        "low_confidence_rate": round(
            sum(1 for r in records if r.low_conf) / len(records), 4
        ),
    }


# ---------------------------------------------------------------------------
# Smoke test / standalone entry point
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Run a few predictions against the loaded model and print results."""
    import pprint

    log.info("Running smoke test …")

    samples = [
        # Low risk — should be no_default
        {
            "age": 45, "income": 95000, "loan_amount": 12000,
            "credit_score": 760, "debt_to_income": 0.18,
            "employment_years": 12.0, "num_credit_lines": 6,
            "missed_payments": 0,
        },
        # High risk — should be default
        {
            "age": 27, "income": 28000, "loan_amount": 18000,
            "credit_score": 520, "debt_to_income": 0.68,
            "employment_years": 0.5, "num_credit_lines": 2,
            "missed_payments": 4,
        },
        # Borderline / low-confidence
        {
            "age": 35, "income": 48000, "loan_amount": 15000,
            "credit_score": 640, "debt_to_income": 0.40,
            "employment_years": 3.5, "num_credit_lines": 4,
            "missed_payments": 1,
        },
    ]

    labels = ["LOW RISK", "HIGH RISK", "BORDERLINE"]

    for label, sample in zip(labels, samples):
        result = predict(sample)
        print(f"\n── {label} ─────────────────────────")
        pprint.pprint({
            "probability":    result.probability,
            "decision":       "DEFAULT" if result.decision == 1 else "NO DEFAULT",
            "low_confidence": result.low_confidence,
            "model_version":  result.model_version,
            "latency_ms":     result.latency_ms,
        })

    print("\n── Batch prediction (3 records) ────")
    batch_results = predict_batch(samples)
    for i, r in enumerate(batch_results):
        print(f"  [{i}] prob={r.probability:.4f}  decision={'DEFAULT' if r.decision else 'NO DEFAULT'}")

    print("\n── Prediction stats ────────────────")
    pprint.pprint(get_prediction_stats())

    log.info("Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _smoke_test()