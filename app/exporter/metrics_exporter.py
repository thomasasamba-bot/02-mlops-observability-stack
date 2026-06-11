"""
app/exporter/metrics_exporter.py

Standalone Prometheus metrics exporter for pipeline-level metrics.

This service runs on port 8007 and exposes metrics that don't belong in
either the anomaly detection service (port 8005) or the inference server
(port 8006). It bridges gaps in the observability stack:

  1. MLflow experiment metrics — pulls latest run metrics from MLflow
     (accuracy, F1, ROC-AUC, training duration) and exposes them as
     Prometheus gauges so Grafana can show model quality over time.

  2. Data pipeline metrics — tracks dataset statistics (row counts,
     default rates, feature means) from the CSV files on disk.

  3. Schema drift baseline — exposes training-set statistics as gauges
     so Grafana can overlay current vs baseline feature distributions.

  4. System health bridge — aggregates health signals from the inference
     server (/health/ready) and MLflow into a single gauge for alerting.

Why a separate exporter?
  The inference server (app/serving/app.py) only has metrics for live
  traffic. MLflow run metrics live in the MLflow store, not in Prometheus.
  This exporter is the bridge — it polls MLflow and the filesystem on a
  configurable interval and publishes the results to Prometheus.

Architecture:
  - FastAPI app with a /metrics endpoint (standard Prometheus scrape target)
  - Background thread polls MLflow + filesystem every POLL_INTERVAL seconds
  - Graceful degradation: if MLflow is unreachable, last known values persist

Endpoints:
  GET /metrics       — Prometheus text exposition
  GET /health/live   — liveness probe
  GET /status        — current metric values (debug)

Usage:
  uvicorn app.exporter.metrics_exporter:app --host 0.0.0.0 --port 8007
  python -m app.exporter.metrics_exporter
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_URI         = os.getenv("MLFLOW_TRACKING_URI",  "http://localhost:5000")
EXPERIMENT_NAME    = os.getenv("MLFLOW_EXPERIMENT",     "credit-scoring")
MODEL_NAME         = os.getenv("MODEL_NAME",             "credit-scoring-model")
SCHEMA_PATH        = Path(os.getenv("SCHEMA_PATH",       "data/processed/feature_schema.json"))
BASELINE_CSV       = Path(os.getenv("BASELINE_CSV",      "data/raw/credit_baseline.csv"))
DRIFTED_CSV        = Path(os.getenv("DRIFTED_CSV",       "data/raw/credit_drifted.csv"))
INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL",     "http://localhost:8006")
POLL_INTERVAL      = int(os.getenv("EXPORTER_POLL_INTERVAL", "60"))
SERVICE_VERSION    = os.getenv("SERVICE_VERSION",        "1.0.0")
LOG_LEVEL          = os.getenv("LOG_LEVEL",              "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Prometheus gauges — MLflow experiment metrics
# ---------------------------------------------------------------------------

MLFLOW_RUN_METRIC = Gauge(
    "mlflow_run_metric",
    "Latest MLflow run metric value",
    ["experiment", "metric"],
)

MLFLOW_MODEL_VERSION = Gauge(
    "mlflow_model_version_registered",
    "Latest registered model version number",
    ["model_name", "stage"],
)

MLFLOW_SCRAPE_ERRORS = Counter(
    "mlflow_scrape_errors_total",
    "Number of times MLflow scraping failed",
)

MLFLOW_LAST_SCRAPE = Gauge(
    "mlflow_last_scrape_timestamp",
    "Unix timestamp of the last successful MLflow scrape",
)

# ---------------------------------------------------------------------------
# Prometheus gauges — dataset statistics
# ---------------------------------------------------------------------------

DATASET_ROWS = Gauge(
    "ml_dataset_rows",
    "Number of rows in a dataset file",
    ["dataset"],    # "baseline" | "drifted"
)

DATASET_DEFAULT_RATE = Gauge(
    "ml_dataset_default_rate",
    "Default (positive label) rate in a dataset",
    ["dataset"],
)

DATASET_FEATURE_MEAN = Gauge(
    "ml_dataset_feature_mean",
    "Mean value of a feature in a dataset",
    ["dataset", "feature"],
)

DATASET_SCRAPE_ERRORS = Counter(
    "ml_dataset_scrape_errors_total",
    "Dataset file scrape errors",
)

# ---------------------------------------------------------------------------
# Prometheus gauges — training schema baselines
# ---------------------------------------------------------------------------

SCHEMA_FEATURE_MEAN = Gauge(
    "ml_schema_feature_mean",
    "Training-set mean for each feature (from feature schema)",
    ["feature"],
)

SCHEMA_FEATURE_STD = Gauge(
    "ml_schema_feature_std",
    "Training-set std for each feature",
    ["feature"],
)

SCHEMA_FEATURE_P50 = Gauge(
    "ml_schema_feature_p50",
    "Training-set median (p50) for each feature",
    ["feature"],
)

SCHEMA_DEFAULT_RATE = Gauge(
    "ml_schema_default_rate",
    "Training-set default rate (from feature schema)",
)

SCHEMA_TRAINING_SAMPLES = Gauge(
    "ml_schema_training_samples",
    "Number of training samples (from feature schema)",
)

# ---------------------------------------------------------------------------
# Prometheus gauges — downstream service health
# ---------------------------------------------------------------------------

INFERENCE_SERVER_UP = Gauge(
    "ml_inference_server_up",
    "1 if inference server /health/live returns 200, 0 otherwise",
)

MLFLOW_SERVER_UP = Gauge(
    "ml_mlflow_server_up",
    "1 if MLflow server is reachable, 0 otherwise",
)

# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _scrape_mlflow() -> None:
    """Pull latest run metrics and model versions from MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        # ── Server health ──────────────────────────────────────────────
        try:
            client.search_experiments(max_results=1)
            MLFLOW_SERVER_UP.set(1)
        except Exception:
            MLFLOW_SERVER_UP.set(0)
            raise

        # ── Latest run metrics ─────────────────────────────────────────
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run = runs[0]
                metrics_of_interest = [
                    "test_roc_auc", "test_f1", "test_accuracy",
                    "test_precision", "test_recall",
                    "cv_roc_auc_mean", "cv_f1_mean",
                    "train_duration_seconds",
                ]
                for metric in metrics_of_interest:
                    val = run.data.metrics.get(metric)
                    if val is not None:
                        MLFLOW_RUN_METRIC.labels(
                            experiment=EXPERIMENT_NAME,
                            metric=metric,
                        ).set(val)
                log.debug(
                    "Scraped MLflow run %s  roc_auc=%.4f",
                    run.info.run_id[:8],
                    run.data.metrics.get("test_roc_auc", 0),
                )

        # ── Model registry versions ────────────────────────────────────
        for stage in ["Staging", "Production", "None"]:
            try:
                versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
                if versions:
                    MLFLOW_MODEL_VERSION.labels(
                        model_name=MODEL_NAME,
                        stage=stage,
                    ).set(int(versions[0].version))
            except Exception:
                pass

        MLFLOW_LAST_SCRAPE.set(time.time())

    except Exception as exc:
        MLFLOW_SCRAPE_ERRORS.inc()
        log.warning("MLflow scrape failed: %s", exc)


def _scrape_datasets() -> None:
    """Read CSV files and update dataset statistics gauges."""
    datasets = {
        "baseline": BASELINE_CSV,
        "drifted":  DRIFTED_CSV,
    }
    features = [
        "age", "income", "loan_amount", "credit_score",
        "debt_to_income", "employment_years", "num_credit_lines", "missed_payments",
    ]

    for name, path in datasets.items():
        if not path.exists():
            log.debug("Dataset not found, skipping: %s", path)
            continue
        try:
            df = pd.read_csv(path)
            DATASET_ROWS.labels(dataset=name).set(len(df))

            if "default" in df.columns:
                DATASET_DEFAULT_RATE.labels(dataset=name).set(
                    float(df["default"].mean())
                )

            for feat in features:
                if feat in df.columns:
                    DATASET_FEATURE_MEAN.labels(
                        dataset=name, feature=feat
                    ).set(float(df[feat].mean()))

            log.debug("Scraped dataset %s  rows=%d", name, len(df))

        except Exception as exc:
            DATASET_SCRAPE_ERRORS.inc()
            log.warning("Dataset scrape failed (%s): %s", name, exc)


def _scrape_schema() -> None:
    """Read feature schema and publish training baseline stats."""
    if not SCHEMA_PATH.exists():
        log.debug("Schema not found, skipping: %s", SCHEMA_PATH)
        return
    try:
        schema = json.loads(SCHEMA_PATH.read_text())
        features = schema.get("features", {})

        for feat, stats in features.items():
            SCHEMA_FEATURE_MEAN.labels(feature=feat).set(stats.get("mean", 0))
            SCHEMA_FEATURE_STD.labels(feature=feat).set(stats.get("std",  0))
            SCHEMA_FEATURE_P50.labels(feature=feat).set(stats.get("p50",  0))

        target = schema.get("target", {})
        if "default_rate" in target:
            SCHEMA_DEFAULT_RATE.set(target["default_rate"])
        if "n_samples" in target:
            SCHEMA_TRAINING_SAMPLES.set(target["n_samples"])

        log.debug("Scraped schema from %s", SCHEMA_PATH)

    except Exception as exc:
        log.warning("Schema scrape failed: %s", exc)


def _scrape_inference_server() -> None:
    """Probe the inference server liveness endpoint."""
    import urllib.request
    import urllib.error

    url = f"{INFERENCE_BASE_URL}/health/live"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            INFERENCE_SERVER_UP.set(1 if resp.status == 200 else 0)
    except Exception:
        INFERENCE_SERVER_UP.set(0)


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

class _MetricsPoller:
    def __init__(self) -> None:
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="metrics-poller", daemon=True,
        )
        self._thread.start()
        log.info("MetricsPoller started  interval=%ds", POLL_INTERVAL)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("MetricsPoller stopped")

    def _loop(self) -> None:
        # Run immediately on start, then every POLL_INTERVAL seconds
        while not self._stop.is_set():
            self._poll()
            self._stop.wait(POLL_INTERVAL)

    def _poll(self) -> None:
        log.debug("Polling metrics …")
        _scrape_mlflow()
        _scrape_datasets()
        _scrape_schema()
        _scrape_inference_server()
        log.debug("Poll complete")


_poller = _MetricsPoller()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting metrics exporter v%s", SERVICE_VERSION)
    _poller.start()
    yield
    _poller.stop()
    log.info("Metrics exporter shutdown complete")


app = FastAPI(
    title="ML Pipeline Metrics Exporter",
    description=(
        "Prometheus exporter for MLflow experiment metrics, dataset statistics, "
        "and training schema baselines."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus text format metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health/live")
async def liveness():
    return JSONResponse({"status": "alive", "service": "metrics-exporter"})


@app.get("/status")
async def status():
    """Current snapshot of all tracked metric values."""
    schema_data: dict = {}
    if SCHEMA_PATH.exists():
        try:
            raw = json.loads(SCHEMA_PATH.read_text())
            schema_data = {
                "features": list(raw.get("features", {}).keys()),
                "default_rate": raw.get("target", {}).get("default_rate"),
                "n_samples":    raw.get("target", {}).get("n_samples"),
            }
        except Exception:
            pass

    datasets: dict = {}
    for name, path in [("baseline", BASELINE_CSV), ("drifted", DRIFTED_CSV)]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                datasets[name] = {
                    "rows": len(df),
                    "default_rate": float(df["default"].mean()) if "default" in df.columns else None,
                }
            except Exception:
                datasets[name] = {"error": "read failed"}
        else:
            datasets[name] = {"error": "file not found"}

    return JSONResponse({
        "service":          "metrics-exporter",
        "version":          SERVICE_VERSION,
        "poll_interval_s":  POLL_INTERVAL,
        "mlflow_uri":       MLFLOW_URI,
        "experiment":       EXPERIMENT_NAME,
        "schema":           schema_data,
        "datasets":         datasets,
        "sources": {
            "mlflow":           MLFLOW_URI,
            "inference_server": INFERENCE_BASE_URL,
            "schema":           str(SCHEMA_PATH),
            "baseline_csv":     str(BASELINE_CSV),
            "drifted_csv":      str(DRIFTED_CSV),
        },
    })


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": "ml-pipeline-metrics-exporter",
        "metrics": "/metrics",
        "health":  "/health/live",
        "status":  "/status",
        "docs":    "/docs",
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.exporter.metrics_exporter:app",
        host="0.0.0.0",
        port=8007,
        workers=1,
        log_level=LOG_LEVEL.lower(),
        reload=False,
    )