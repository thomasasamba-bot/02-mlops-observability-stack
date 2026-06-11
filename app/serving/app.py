"""
app/serving/app.py

FastAPI ML inference server — port 8006.

Responsibilities:
  1. Serve credit scoring predictions via REST API
  2. Start the DriftDetector background thread on startup
  3. Expose /health/live, /health/ready, /metrics (Prometheus), /drift/status
  4. Log prediction confidence and latency to Prometheus
  5. Route low-confidence predictions with a flag for human review

Endpoints:
  POST /predict              — single record inference
  POST /predict/batch        — batch inference (up to 500 records)
  GET  /drift/status         — latest drift report
  GET  /drift/report         — full per-feature drift breakdown
  GET  /health/live          — liveness (process is up)
  GET  /health/ready         — readiness (model loaded, drift detector running)
  GET  /metrics              — Prometheus text exposition
  GET  /status               — service status + prediction stats

Design:
  - Model loaded lazily on first request (predict.py singleton)
  - DriftDetector started in FastAPI lifespan (clean startup/shutdown)
  - All Prometheus metrics defined in predict.py and drift_detector.py;
    this file only adds request-level counters/histograms
  - CORS enabled for Grafana and local dashboard access

Usage:
  uvicorn app.serving.app:app --host 0.0.0.0 --port 8006 --workers 1
  (workers=1 required — model cache and prediction buffer are in-process)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVICE_NAME    = os.getenv("SERVICE_NAME",    "ml-inference-server")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE", "500"))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Request-level Prometheus metrics (service layer, not model layer)
# ---------------------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

BATCH_SIZE_HIST = Histogram(
    "ml_batch_size",
    "Batch prediction sizes",
    buckets=[1, 5, 10, 25, 50, 100, 200, 500],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    age:              float = Field(..., ge=18,  le=80,      description="Applicant age")
    income:           float = Field(..., ge=0,               description="Annual income USD")
    loan_amount:      float = Field(..., ge=0,               description="Requested loan USD")
    credit_score:     float = Field(..., ge=300, le=850,     description="FICO credit score")
    debt_to_income:   float = Field(..., ge=0.0, le=1.0,     description="Debt-to-income ratio")
    employment_years: float = Field(..., ge=0,               description="Years at current employer")
    num_credit_lines: float = Field(..., ge=0,               description="Open credit lines")
    missed_payments:  float = Field(..., ge=0,               description="Missed payments (24 months)")

    @field_validator("income", "loan_amount")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    def to_dict(self) -> dict[str, float]:
        return self.model_dump()


class PredictResponse(BaseModel):
    probability:    float
    decision:       str          # "DEFAULT" | "NO_DEFAULT"
    low_confidence: bool
    model_version:  str
    latency_ms:     float
    review_required: bool        # True when low_confidence=True


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest] = Field(..., min_length=1)

    @field_validator("records")
    @classmethod
    def check_size(cls, v: list) -> list:
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum {MAX_BATCH_SIZE}")
        return v


class BatchPredictResponse(BaseModel):
    predictions:    list[PredictResponse]
    total:          int
    defaults:       int
    no_defaults:    int
    low_confidence: int
    latency_ms:     float


class HealthResponse(BaseModel):
    status:  str
    service: str
    version: str


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

_drift_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start DriftDetector on startup; stop it on shutdown."""
    global _drift_detector

    log.info("Starting %s v%s", SERVICE_NAME, SERVICE_VERSION)

    # Import here to catch import errors at startup, not at first request
    try:
        from app.pipeline.drift_detector import DriftDetector
        _drift_detector = DriftDetector()
        _drift_detector.start()
        log.info("DriftDetector background thread started")
    except FileNotFoundError as exc:
        log.warning("DriftDetector not started: %s", exc)
        log.warning("Run app/pipeline/train.py to generate the feature schema")
    except Exception as exc:
        log.error("DriftDetector failed to start: %s", exc)

    yield  # application runs here

    # Shutdown
    if _drift_detector:
        _drift_detector.stop()
    log.info("%s shutdown complete", SERVICE_NAME)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Inference Server",
    description=(
        "Credit scoring model inference with real-time drift detection. "
        "Part of the AIOps / MLOps Observability Platform."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — request timing + counters
# ---------------------------------------------------------------------------

@app.middleware("http")
async def track_requests(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    latency  = time.perf_counter() - t0

    # Normalise path (strip dynamic segments for cardinality safety)
    path = request.url.path

    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=path,
        status_code=str(response.status_code),
    ).inc()
    HTTP_LATENCY.labels(endpoint=path).observe(latency)

    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predict_response(result) -> PredictResponse:
    return PredictResponse(
        probability=round(result.probability, 6),
        decision="DEFAULT" if result.decision == 1 else "NO_DEFAULT",
        low_confidence=result.low_confidence,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
        review_required=result.low_confidence,
    )


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single record credit default prediction",
    tags=["Inference"],
)
async def predict(request: PredictRequest):
    """
    Score a single credit application.

    Returns the default probability, binary decision, and a
    `review_required` flag for low-confidence predictions (probability
    in the 0.35–0.55 band) that should be routed for human review.
    """
    from app.pipeline.predict import predict as _predict, ValidationError

    try:
        result = _predict(request.to_dict())
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc))
    except RuntimeError as exc:
        log.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Model not available: {exc}")
    except Exception as exc:
        log.exception("Unexpected prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))

    return _make_predict_response(result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch credit default predictions",
    tags=["Inference"],
)
async def predict_batch(request: BatchPredictRequest):
    """
    Score a batch of credit applications (max 500 records).

    More efficient than calling /predict in a loop — single model
    forward pass for the entire batch.
    """
    from app.pipeline.predict import predict_batch as _predict_batch, ValidationError

    t0      = time.perf_counter()
    records = [r.to_dict() for r in request.records]
    BATCH_SIZE_HIST.observe(len(records))

    try:
        results = _predict_batch(records)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Model not available: {exc}")
    except Exception as exc:
        log.exception("Batch prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc))

    latency_ms   = (time.perf_counter() - t0) * 1000
    predictions  = [_make_predict_response(r) for r in results]
    defaults     = sum(1 for p in predictions if p.decision == "DEFAULT")
    low_conf     = sum(1 for p in predictions if p.low_confidence)

    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        defaults=defaults,
        no_defaults=len(predictions) - defaults,
        low_confidence=low_conf,
        latency_ms=round(latency_ms, 3),
    )


# ---------------------------------------------------------------------------
# Drift endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/drift/status",
    summary="Latest drift detection status",
    tags=["Observability"],
)
async def drift_status():
    """
    Returns whether drift is currently detected and which features are affected.
    Lightweight — suitable for alerting and dashboards polling frequently.
    """
    if _drift_detector is None:
        return JSONResponse({"drift_detector": "not_started",
                             "overall_drift": False})

    report = _drift_detector.get_last_report()
    if report is None:
        return JSONResponse({"drift_detector": "running",
                             "overall_drift": False,
                             "message": "No check completed yet"})

    return JSONResponse({
        "drift_detector":    "running",
        "overall_drift":     report.overall_drift,
        "drifted_features":  report.drifted_features,
        "window_size":       report.window_size,
        "confidence_mean":   round(report.confidence_mean, 4),
        "confidence_std":    round(report.confidence_std,  4),
        "default_rate":      round(report.default_rate,    4),
        "checked_at":        report.timestamp,
        "error":             report.error,
    })


@app.get(
    "/drift/report",
    summary="Full per-feature drift breakdown",
    tags=["Observability"],
)
async def drift_report():
    """
    Full drift report including per-feature PSI, KS-statistic, and severity.
    Used by Grafana for the feature drift heatmap panel.
    """
    if _drift_detector is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="DriftDetector not started")

    report = _drift_detector.get_last_report()
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No drift report available yet")

    return JSONResponse({
        "timestamp":       report.timestamp,
        "window_size":     report.window_size,
        "overall_drift":   report.overall_drift,
        "drifted_features": report.drifted_features,
        "confidence": {
            "mean": round(report.confidence_mean, 4),
            "std":  round(report.confidence_std,  4),
        },
        "default_rate": round(report.default_rate, 4),
        "features": [
            {
                "feature":   r.feature,
                "psi":       round(r.psi, 6),
                "ks_stat":   round(r.ks_stat, 6),
                "ks_pvalue": round(r.ks_pvalue, 6),
                "severity":  r.severity,
                "drifted":   r.severity == "alert",
            }
            for r in report.feature_results
        ],
    })


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Health"],
)
async def liveness():
    """Process is alive. Returns 200 always (if the process is up, it responds)."""
    return HealthResponse(
        status="alive",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    tags=["Health"],
)
async def readiness():
    """
    Ready when the model is loaded and DriftDetector is running.
    Returns 503 if the model hasn't been loaded yet.
    """
    from app.pipeline.predict import _cache

    if _cache.model is None:
        # Trigger lazy load
        try:
            _cache.ensure_loaded()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not ready: {exc}",
            )

    if _cache.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    return HealthResponse(
        status="ready",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/metrics",
    summary="Prometheus metrics",
    tags=["Observability"],
    response_class=PlainTextResponse,
)
async def metrics():
    """Prometheus text format metrics for scraping."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/status",
    summary="Service status and prediction statistics",
    tags=["Observability"],
)
async def service_status():
    """
    Combined service status: model info, prediction stats, drift summary.
    Used by the ops dashboard.
    """
    from app.pipeline.predict import get_prediction_stats, _cache

    drift_summary: dict[str, Any] = {"detector": "not_started"}
    if _drift_detector:
        report = _drift_detector.get_last_report()
        drift_summary = {
            "detector":      "running",
            "overall_drift": report.overall_drift if report else False,
            "drifted_features": report.drifted_features if report else [],
            "last_check":    report.timestamp if report else None,
        }

    return JSONResponse({
        "service":  SERVICE_NAME,
        "version":  SERVICE_VERSION,
        "model": {
            "version": _cache.version,
            "loaded":  _cache.model is not None,
        },
        "predictions": get_prediction_stats(),
        "drift":       drift_summary,
    })


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs":    "/docs",
        "health":  "/health/live",
        "metrics": "/metrics",
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.serving.app:app",
        host="0.0.0.0",
        port=8006,
        workers=1,       # in-process model cache requires single worker
        log_level=LOG_LEVEL.lower(),
        reload=False,
    )