# Project : MLOps Observability Platform

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-119%20passed-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-3.13.0-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-validated-326CE5?logo=kubernetes&logoColor=white)
![CI](https://github.com/thomasasamba-bot/02-mlops-observability-stack/actions/workflows/ci.yml/badge.svg)

> **See inside your ML system — not just around it.**

**`02-mlops-observability-stack`** is a production-ready observability platform that transforms blind spots into actionable insights. It detects model drift before predictions fail, tracks feature distribution shifts in real-time, and exposes ML pipeline health as native Prometheus metrics — all without manual intervention.

Combines infrastructure-layer anomaly detection (CPU/memory/disk via Z-Score, EWMA, Isolation Forest) with deep ML pipeline visibility (PSI + KS-test drift scoring, prediction confidence tracking, feature stability monitoring). Built for SRE and AIops/Mlops teams running credit scoring, fraud detection, and other high-stakes AI/ML systems.

Part of a public SRE/AIOps portfolio by **Thomas Asamba** — Senior SRE and Cloud/DevOps Engineer, Nairobi.

🔗 [Project 1: AIOps Self-Healing Infrastructure](https://github.com/thomasasamba-bot/01-aiops-self-healing-infrastructure)

---

## Documentation

Comprehensive guides and references for using this platform:

| Document | Description |
|----------|-------------|
| **[CI/CD Pipeline Guide](docs/guides/ci-cd-pipeline.md)** | GitHub Actions workflows, deployment process, debugging |
| **[Configuration Reference](docs/guides/configuration.md)** | All environment variables and settings |
| **[API Reference](docs/api/endpoints.md)** | Complete endpoint documentation with examples |
| **[Observability Setup](docs/observability/grafana-setup.md)** | Grafana dashboards, Prometheus scraping, monitoring |
| **[Metrics Reference](docs/observability/metrics-reference.md)** | Full metric catalog, queries, and retention |
| **[Alert Rules & Runbooks](docs/observability/alerts-rules.md)** | Alert definitions, escalation, remediation |
| **[Troubleshooting](docs/guides/troubleshooting.md)** | Common issues and solutions |
| **[Development Guide](docs/dev/contributing.md)** | Local setup, coding standards, testing |

---

## The problem this solves

Standard infrastructure monitoring tells you *that* a service is slow or erroring. It cannot tell you *why* the model started making worse predictions last Tuesday. For that you need to observe the model itself:

- Are the features being scored today drawn from the same distribution the model was trained on?
- Is the model's confidence degrading — probabilities clustering toward 0.5?
- Has the default rate in live predictions shifted relative to the training baseline?

This platform answers those questions continuously, in production, with zero manual intervention.

---

## Architecture

![MLOps Observability Platform Architecture](docs/architecture/architecture-overview.png)

The platform has two observability layers. The infrastructure layer (Layer 1) watches CPU, memory, disk, and network using Z-Score, EWMA, and Isolation Forest. The ML pipeline layer (Layer 2) watches what's happening *inside* the model — feature distributions, prediction confidence, and drift scores.

### Metrics pipeline
![Metrics Pipeline](docs/architecture/METRICS_PIPELINE-Time_Series_Observability.png)

### Logs pipeline
![Logs Pipeline](docs/architecture/LOGS_PIPELINE-Event_Driven_Observability.png)

---

## What's inside

### ML Pipeline (`app/pipeline/`)

| File | Purpose |
|------|---------|
| `train.py` | RandomForestClassifier with full MLflow tracking — params, metrics, feature importances, model registration |
| `predict.py` | Lazy-loaded model singleton, rolling prediction buffer, confidence flagging, thread-safe batch inference |
| `drift_detector.py` | PSI + KS-test drift detection; frequency-table PSI for discrete features; background thread with configurable interval |

### Serving Layer (`app/serving/`)

FastAPI inference server on port 8006. Single-worker (intentional — in-process prediction buffer). Endpoints:

```
POST /predict              Single credit scoring prediction
POST /predict/batch        Batch predictions (up to 500 records)
GET  /drift/status         Latest drift detection status
GET  /drift/report         Full per-feature PSI/KS breakdown
GET  /health/live          Liveness probe
GET  /health/ready         Readiness probe (model loaded check)
GET  /metrics              Prometheus text exposition
GET  /status               Service status + prediction stats
```

### Metrics Exporter (`app/exporter/`)

Standalone Prometheus exporter on port 8007. Bridges MLflow experiment metrics (ROC-AUC, F1, training duration) and dataset statistics into Prometheus so Grafana can show model quality over time without scraping MLflow directly.

### Infrastructure Anomaly Detection (`app/anomaly_detection/`)

Z-Score, EWMA, and Isolation Forest on CPU/memory/disk/network metrics. Composite weighted scoring with Alertmanager integration. Separate from the ML pipeline layer — watches *around* the model, not inside it.

---

## Drift detection design

Two complementary methods run on every cycle:

**PSI (Population Stability Index)**
Measures how much the current feature distribution has shifted from the training baseline. Uses training-set percentile bins as the reference — so the baseline proportion is exactly `1/n_bins` per bin by construction, and only the current data needs to be binned.

For integer-valued features (`missed_payments`, `num_credit_lines`) where percentile bins degenerate, frequency-table PSI is used instead — comparing empirical `P(X=k)` from training against the live distribution.

| PSI | Interpretation |
|-----|---------------|
| < 0.10 | Stable — no action needed |
| 0.10–0.25 | Moderate shift — monitor |
| > 0.25 | Significant drift — investigate or retrain |

**KS-test**
Non-parametric test for whether two samples come from the same distribution. `p < 0.05` → statistically significant shift. Can detect subtle distributional changes that PSI misses (e.g. shape changes without mean shift).

**Confidence degradation tracking**
As drift increases, model probabilities cluster toward 0.5 — the model becomes uncertain. `ml_prediction_confidence_mean` dropping below 0.40 is a leading indicator of accuracy degradation, visible in Grafana before ROC-AUC degrades in MLflow.

---

## Test results

> Unit tests run fully offline — no MLflow server or inference server required.
> Integration tests require all four services running (MLflow, inference server, metrics exporter, and at least one completed drift check cycle).

```
tests/unit/test_training.py        24 passed   7.3s   (offline)
tests/unit/test_predict.py         30 passed   9.5s   (offline)
tests/unit/test_drift_detector.py  32 passed   5.4s   (offline)
tests/integration/test_pipeline.py 33 passed  52.0s   (requires live services)
─────────────────────────────────────────────────────────────────
Total                              119 passed  74.2s
```

**Chaos test** (`tests/chaos/inject_drift.py`) validates the full signal chain end-to-end — from drifted CSV → HTTP batch requests → prediction buffer → background drift check → `/drift/status` API response:

```
Injecting 300 drifted records  → drift detected after  2 polls  (20s)
Injecting 300 baseline records → drift clears  after 11 polls (110s)
```

Confidence mean shift during chaos test: `0.595` (drifted) → `0.314` (stable) — model uncertainty as a leading indicator.

---

## Quick start

**Prerequisites:** Python 3.12+, Docker Desktop, minikube (optional)

```bash
# 1. Clone and set up
git clone https://github.com/thomasasamba-bot/02-mlops-observability-stack
cd 02-mlops-observability-stack
bash scripts/bootstrap/setup.sh

# 2. Start MLflow (Terminal 1)
source .venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000

# 3. Train the model (Terminal 2)
source .venv/bin/activate
python -m app.pipeline.train

# 4. Start inference server (Terminal 3)
source .venv/bin/activate
uvicorn app.serving.app:app --host 0.0.0.0 --port 8006 --workers 1

# 5. Start metrics exporter (Terminal 4)
source .venv/bin/activate
uvicorn app.exporter.metrics_exporter:app --host 0.0.0.0 --port 8007

# 6. Test a prediction
curl -s -X POST http://localhost:8006/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "income": 95000, "loan_amount": 12000,
    "credit_score": 760, "debt_to_income": 0.18,
    "employment_years": 12.0, "num_credit_lines": 6,
    "missed_payments": 0
  }' | python3 -m json.tool

# 7. Run the chaos test (inject drifted traffic)
python tests/chaos/inject_drift.py

# 8. Check drift status
curl -s http://localhost:8006/drift/status | python3 -m json.tool
```

**Full stack with Docker Compose:**
```bash
bash scripts/deployment/deploy-local.sh --build --train
```

**Deploy to Kubernetes (minikube):**
```bash
eval $(minikube docker-env)
bash scripts/deployment/deploy-k8s.sh --build --train
```

---

## Model performance

Trained on a synthetic credit scoring dataset (5,000 records, ~25% default rate):

| Metric | Value |
|--------|-------|
| CV ROC-AUC | 0.848 ± 0.011 |
| Test ROC-AUC | 0.829 |
| Test F1 | 0.682 |
| Test Accuracy | 0.850 |
| Decision threshold | 0.40 (recall-focused) |

Top features by importance: `credit_score` (0.23), `income` (0.21), `loan_amount` (0.15), `debt_to_income` (0.14).

---

## Prometheus Metrics Overview

The platform exports metrics across three dimensions: **drift detection**, **prediction quality**, and **model performance**.

| Metric Category | Key Metrics | Description |
|-----------------|------------|-------------|
| **Drift Detection** | `ml_drift_detected`, `ml_feature_drift_psi{feature}`, `ml_feature_drift_ks_pvalue{feature}` | Feature distribution shift (PSI > 0.25 = alert) |
| **Prediction Health** | `ml_prediction_confidence_mean`, `ml_prediction_default_rate`, `ml_predictions_total{decision}` | Model confidence and decision breakdown |
| **Model Quality** | `mlflow_run_metric{metric="test_roc_auc"}`, `mlflow_run_metric{metric="test_f1"}` | Latest trained model metrics |
| **Infrastructure** | `infrastructure_cpu_zscore`, `infrastructure_memory_ewma`, `infrastructure_disk_isolation_forest` | Anomaly detection on CPU/memory/disk |

**Full reference:** [Metrics Documentation](docs/observability/metrics-reference.md) — includes Prometheus queries and dashboard examples.

---

## Alert Rules

The platform fires alerts when metrics exceed operational thresholds:

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `ModelDriftDetected` | PSI > 0.25 for 2m | ⚠️ Warning | Check feature distributions |
| `PredictionConfidenceLow` | mean confidence < 0.35 for 5m | ⚠️ Warning | Model uncertainty increasing |
| `ModelAccuracyDegraded` | ROC-AUC < 0.75 | 🔴 Critical | Investigate or retrain |
| `HighDefaultRate` | default_rate > 50% for 10m | ⚠️ Warning | Portfolio risk shift |
| `InferenceServerDown` | server unreachable for 1m | 🔴 Critical | Restore server |

**Full rules & runbooks:** [Alert Documentation](docs/runbooks/) — includes thresholds, escalation, and remediation steps.

---

## Project structure

![Project folder structure](docs/diagrams/project-folder-structure-overview.png)

---

## Related Projects

- [01-aiops-self-healing-infrastructure](https://github.com/thomasasamba-bot/01-aiops-self-healing-infrastructure) — AIOps self-healing with Lambda/SSM
- [03-secure-aws-infrastructure](https://github.com/thomasasamba-bot/03-secure-aws-infrastructure) — IaC with KMS and IAM hardening
- [04-kubernetes-orchestration](https://github.com/thomasasamba-bot/04-kubernetes-orchestration) — EKS zero-downtime deployments
- [05-devsecops-pipeline](https://github.com/thomasasamba-bot/05-devsecops-pipeline) — CI/CD with SonarQube and Trivy

---

*Built by [Thomas Asamba](https://linkedin.com/in/thomasasamba) | [github.com/thomasasamba-bot](https://github.com/thomasasamba-bot)*