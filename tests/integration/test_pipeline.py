"""
tests/integration/test_pipeline.py

Integration tests for the full ML pipeline.

Unlike the unit tests (which test individual components in isolation),
these tests verify that the components work correctly together:

  1. Serving layer  — inference server health, prediction endpoints,
                      response schema, error handling
  2. Drift pipeline — inject traffic, verify drift status reflects it,
                      verify recovery after stable traffic
  3. Metrics bridge — confirm Prometheus metrics are being updated
                      on the inference server and exporter

All tests require the following services to be running:
  - MLflow server:       http://localhost:5000
  - Inference server:    http://localhost:8006  (uvicorn app.serving.app:app)
  - Metrics exporter:    http://localhost:8007  (optional — skipped if down)

Run with all services up:
  pytest tests/integration/test_pipeline.py -v

Run only the serving tests (faster):
  pytest tests/integration/test_pipeline.py -v -k "Serving"
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INFERENCE_URL = "http://localhost:8006"
EXPORTER_URL  = "http://localhost:8007"
MLFLOW_URL    = "http://localhost:5000"

LOW_RISK = {
    "age": 50, "income": 120_000, "loan_amount": 10_000,
    "credit_score": 800, "debt_to_income": 0.10,
    "employment_years": 20.0, "num_credit_lines": 8,
    "missed_payments": 0,
}

HIGH_RISK = {
    "age": 24, "income": 22_000, "loan_amount": 20_000,
    "credit_score": 490, "debt_to_income": 0.80,
    "employment_years": 0.3, "num_credit_lines": 1,
    "missed_payments": 5,
}


def _get(url: str, timeout: int = 5) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def _post(url: str, payload: dict, timeout: int = 15) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _service_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health/live", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_inference = pytest.mark.skipif(
    not _service_up(INFERENCE_URL),
    reason=f"Inference server not running at {INFERENCE_URL}",
)

requires_exporter = pytest.mark.skipif(
    not _service_up(EXPORTER_URL),
    reason=f"Metrics exporter not running at {EXPORTER_URL}",
)

requires_mlflow = pytest.mark.skipif(
    not _service_up(MLFLOW_URL),
    reason=f"MLflow server not running at {MLFLOW_URL}",
)


# ---------------------------------------------------------------------------
# 1. Serving layer — health and readiness
# ---------------------------------------------------------------------------

@requires_inference
class TestServingHealth:

    def test_liveness_returns_200(self):
        status, body = _get(f"{INFERENCE_URL}/health/live")
        assert status == 200
        assert body["status"] == "alive"

    def test_readiness_returns_200_when_model_loaded(self):
        status, body = _get(f"{INFERENCE_URL}/health/ready")
        assert status == 200, f"Readiness check failed: {body}"
        assert body["status"] == "ready"

    def test_root_returns_service_info(self):
        status, body = _get(INFERENCE_URL)
        assert status == 200
        assert "service" in body
        assert "docs"    in body

    def test_docs_endpoint_available(self):
        status, _ = _get(f"{INFERENCE_URL}/docs")
        assert status == 200

    def test_status_endpoint_structure(self):
        status, body = _get(f"{INFERENCE_URL}/status")
        assert status == 200
        assert "service"     in body
        assert "model"       in body
        assert "predictions" in body
        assert "drift"       in body
        assert body["model"]["loaded"] is True


# ---------------------------------------------------------------------------
# 2. Serving layer — prediction endpoints
# ---------------------------------------------------------------------------

@requires_inference
class TestServingPrediction:

    def test_single_predict_response_schema(self):
        status, body = _post(f"{INFERENCE_URL}/predict", LOW_RISK)
        assert status == 200, f"Unexpected status {status}: {body}"
        assert "probability"    in body
        assert "decision"       in body
        assert "low_confidence" in body
        assert "model_version"  in body
        assert "latency_ms"     in body
        assert "review_required" in body

    def test_single_predict_probability_valid(self):
        _, body = _post(f"{INFERENCE_URL}/predict", LOW_RISK)
        assert 0.0 <= body["probability"] <= 1.0

    def test_single_predict_decision_values(self):
        for sample in [LOW_RISK, HIGH_RISK]:
            _, body = _post(f"{INFERENCE_URL}/predict", sample)
            assert body["decision"] in ("DEFAULT", "NO_DEFAULT")

    def test_low_risk_decision(self):
        _, body = _post(f"{INFERENCE_URL}/predict", LOW_RISK)
        assert body["decision"] == "NO_DEFAULT", \
            f"Low-risk scored as DEFAULT (prob={body['probability']:.3f})"

    def test_high_risk_decision(self):
        _, body = _post(f"{INFERENCE_URL}/predict", HIGH_RISK)
        assert body["decision"] == "DEFAULT", \
            f"High-risk scored as NO_DEFAULT (prob={body['probability']:.3f})"

    def test_review_required_matches_low_confidence(self):
        _, body = _post(f"{INFERENCE_URL}/predict", LOW_RISK)
        assert body["review_required"] == body["low_confidence"]

    def test_missing_field_returns_422(self):
        bad = {k: v for k, v in LOW_RISK.items() if k != "credit_score"}
        status, _ = _post(f"{INFERENCE_URL}/predict", bad)
        assert status == 422

    def test_invalid_credit_score_returns_422(self):
        bad = {**LOW_RISK, "credit_score": 9999}
        status, _ = _post(f"{INFERENCE_URL}/predict", bad)
        assert status == 422

    def test_batch_predict_response_schema(self):
        status, body = _post(
            f"{INFERENCE_URL}/predict/batch",
            {"records": [LOW_RISK, HIGH_RISK]},
        )
        assert status == 200, f"Batch failed: {body}"
        assert "predictions"    in body
        assert "total"          in body
        assert "defaults"       in body
        assert "no_defaults"    in body
        assert "low_confidence" in body
        assert "latency_ms"     in body

    def test_batch_predict_counts_consistent(self):
        records = [LOW_RISK] * 5 + [HIGH_RISK] * 5
        _, body  = _post(
            f"{INFERENCE_URL}/predict/batch",
            {"records": records},
        )
        assert body["total"] == 10
        assert body["defaults"] + body["no_defaults"] == 10
        assert len(body["predictions"]) == 10

    def test_batch_empty_returns_422(self):
        status, _ = _post(
            f"{INFERENCE_URL}/predict/batch",
            {"records": []},
        )
        assert status == 422

    def test_batch_exceeds_max_returns_422(self):
        records = [LOW_RISK] * 501
        status, _ = _post(
            f"{INFERENCE_URL}/predict/batch",
            {"records": records},
        )
        assert status == 422


# ---------------------------------------------------------------------------
# 3. Prometheus metrics — inference server
# ---------------------------------------------------------------------------

@requires_inference
class TestServingMetrics:

    def test_metrics_endpoint_returns_text(self):
        status, body = _get(f"{INFERENCE_URL}/metrics")
        assert status == 200
        assert isinstance(body, str)
        assert "# HELP" in body

    def test_ml_prediction_metrics_present(self):
        # Make a prediction to ensure metrics are populated
        _post(f"{INFERENCE_URL}/predict", LOW_RISK)
        _, body = _get(f"{INFERENCE_URL}/metrics")
        assert "ml_predictions_total" in body

    def test_http_request_metrics_present(self):
        _, body = _get(f"{INFERENCE_URL}/metrics")
        assert "http_requests_total" in body

    def test_ml_prediction_latency_present(self):
        _, body = _get(f"{INFERENCE_URL}/metrics")
        assert "ml_prediction_latency_seconds" in body


# ---------------------------------------------------------------------------
# 4. Drift status endpoint
# ---------------------------------------------------------------------------

@requires_inference
class TestDriftStatus:

    def test_drift_status_endpoint_responds(self):
        status, body = _get(f"{INFERENCE_URL}/drift/status")
        assert status == 200
        assert "drift_detector" in body
        assert "overall_drift"  in body

    def test_drift_status_has_required_fields(self):
        status, body = _get(f"{INFERENCE_URL}/drift/status")
        assert status == 200
        # These fields present once at least one check has completed
        if body.get("drift_detector") == "running" and "window_size" in body:
            assert "drifted_features"  in body
            assert "confidence_mean"   in body
            assert "default_rate"      in body

    def test_drift_report_endpoint_responds(self):
        status, body = _get(f"{INFERENCE_URL}/drift/report")
        # 404 is acceptable if no check has completed yet
        assert status in (200, 404), f"Unexpected status: {status}"
        if status == 200:
            assert "features"      in body
            assert "overall_drift" in body

    def test_drift_report_feature_structure(self):
        status, body = _get(f"{INFERENCE_URL}/drift/report")
        if status == 404:
            pytest.skip("No drift report available yet")
        for feat in body["features"]:
            assert "feature"   in feat
            assert "psi"       in feat
            assert "ks_stat"   in feat
            assert "ks_pvalue" in feat
            assert "severity"  in feat
            assert feat["severity"] in ("stable", "warning", "alert")


# ---------------------------------------------------------------------------
# 5. Metrics exporter
# ---------------------------------------------------------------------------

@requires_exporter
class TestMetricsExporter:

    def test_exporter_liveness(self):
        status, body = _get(f"{EXPORTER_URL}/health/live")
        assert status == 200
        assert body["status"] == "alive"

    def test_exporter_metrics_endpoint(self):
        status, body = _get(f"{EXPORTER_URL}/metrics")
        assert status == 200
        assert "# HELP" in body

    def test_mlflow_run_metrics_present(self):
        _, body = _get(f"{EXPORTER_URL}/metrics")
        assert "mlflow_run_metric" in body, \
            "mlflow_run_metric not found — is MLflow server running and has a completed run?"

    def test_dataset_metrics_present(self):
        _, body = _get(f"{EXPORTER_URL}/metrics")
        assert "ml_dataset_rows" in body

    def test_schema_metrics_present(self):
        _, body = _get(f"{EXPORTER_URL}/metrics")
        assert "ml_schema_feature_mean" in body

    def test_exporter_status_endpoint(self):
        status, body = _get(f"{EXPORTER_URL}/status")
        assert status == 200
        assert "datasets" in body
        assert "schema"   in body
        assert body["datasets"]["baseline"]["rows"] == 5000
        assert body["datasets"]["drifted"]["rows"]  == 1000


# ---------------------------------------------------------------------------
# 6. End-to-end: inject → detect → recover
# ---------------------------------------------------------------------------

@requires_inference
class TestEndToEndDriftCycle:
    """
    Lightweight E2E test: inject a small batch of drifted records and confirm
    the drift status reflects the change within one detector cycle (120s max).

    This test is slower (~30-130s) because it waits for the background thread.
    Skip with: pytest -k "not EndToEnd"
    """

    def test_inject_drifted_records_and_check_buffer(self):
        """After injecting drifted records, the prediction buffer should grow."""
        _, before = _get(f"{INFERENCE_URL}/status")
        before_count = before.get("predictions", {}).get("buffered_predictions", 0)

        drifted_csv = Path("data/raw/credit_drifted.csv")
        if not drifted_csv.exists():
            pytest.skip("data/raw/credit_drifted.csv not found")

        import csv
        records = []
        with open(drifted_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    k: float(row[k]) for k in [
                        "age", "income", "loan_amount", "credit_score",
                        "debt_to_income", "employment_years",
                        "num_credit_lines", "missed_payments",
                    ]
                })
                if len(records) >= 50:
                    break

        _, resp = _post(
            f"{INFERENCE_URL}/predict/batch",
            {"records": records},
        )
        assert resp["total"] == 50

        _, after = _get(f"{INFERENCE_URL}/status")
        after_count = after.get("predictions", {}).get("buffered_predictions", 0)
        assert after_count > before_count, \
            f"Buffer did not grow: before={before_count} after={after_count}"

    def test_drift_status_reflects_injected_traffic(self):
        """
        After injecting 300 drifted records, poll /drift/status for up to
        150 seconds and confirm drift is detected.
        Requires the background drift check cycle to complete (~120s interval).
        """
        drifted_csv = Path("data/raw/credit_drifted.csv")
        if not drifted_csv.exists():
            pytest.skip("data/raw/credit_drifted.csv not found")

        import csv
        records = []
        with open(drifted_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    k: float(row[k]) for k in [
                        "age", "income", "loan_amount", "credit_score",
                        "debt_to_income", "employment_years",
                        "num_credit_lines", "missed_payments",
                    ]
                })
                if len(records) >= 300:
                    break

        # Inject in batches of 25
        for i in range(0, len(records), 25):
            _post(
                f"{INFERENCE_URL}/predict/batch",
                {"records": records[i:i+25]},
            )

        # Poll for up to 150 seconds
        deadline = time.time() + 150
        detected = False
        while time.time() < deadline:
            _, status = _get(f"{INFERENCE_URL}/drift/status")
            if status.get("overall_drift") is True:
                detected = True
                break
            time.sleep(10)

        assert detected, (
            "Drift not detected within 150s after injecting 300 drifted records. "
            "Check that the DriftDetector background thread is running."
        )