# Project 2: MLOps Observability Platform

> **Observe what's happening *inside* your ML system — not just around it.**

A production-grade MLOps observability stack that combines infrastructure anomaly detection with deep ML pipeline visibility: model drift scores, prediction confidence distributions, and feature shift detection exposed as Prometheus metrics and visualised in Grafana.

Part of a public SRE/AIOps portfolio by **Thomas Asamba** — Senior SRE and Cloud/DevOps Engineer, Nairobi.

🔗 [Project 1: AIOps Self-Healing Infrastructure](https://github.com/thomasasamba-bot/01-aiops-self-healing-infrastructure)

---

## The problem this solves

Standard infrastructure monitoring tells you *that* a service is slow or erroring. It cannot tell you *why* the model started making worse predictions last Tuesday. For that you need to observe the model itself:

- Are the features being scored today drawn from the same distribution the model was trained on?
- Is the model's confidence degrading — probabilities clustering toward 0.5?
- Has the default rate in live predictions shifted relative to the training baseline?

This platform answers those questions continuously, in production, with zero manual intervention.

---

## Architecture

![MLOps Observability Platform Architecture](docs/architecture/architecture-overview.svg)

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

```
tests/unit/test_training.py        24 passed   7.3s
tests/unit/test_predict.py         30 passed   9.5s
tests/unit/test_drift_detector.py  32 passed   5.4s
tests/integration/test_pipeline.py 33 passed  52.0s
─────────────────────────────────────────────────
Total                              119 passed  74.2s
```

The chaos test (`tests/chaos/inject_drift.py`) validates the full signal chain end-to-end:

```
Injecting 300 drifted records → drift detected after 2 polls (20s)
Injecting 300 baseline records → drift clears after 11 polls (110s)
```

---

## Quick start

**Prerequisites:** Python 3.12+, Docker Desktop, minikube (optional)

```bash
# 1. Clone and set up
git clone https://github.com/thomasasamba-bot/02-observability-ml-alerting
cd 02-observability-ml-alerting
bash scripts/bootstrap/setup.sh

# 2. Start MLflow (Terminal 1)
mlflow server --host 0.0.0.0 --port 5000

# 3. Train the model (Terminal 2)
source .venv/bin/activate
python -m app.pipeline.train

# 4. Start inference server (Terminal 3)
uvicorn app.serving.app:app --host 0.0.0.0 --port 8006 --workers 1

# 5. Start metrics exporter (Terminal 4)
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

## Prometheus metrics reference

| Metric | Type | Description |
|--------|------|-------------|
| `ml_feature_drift_psi{feature}` | Gauge | PSI per feature vs training baseline |
| `ml_feature_drift_ks_statistic{feature}` | Gauge | KS-test statistic per feature |
| `ml_feature_drift_ks_pvalue{feature}` | Gauge | KS-test p-value per feature |
| `ml_drift_detected` | Gauge | 1 if any feature PSI > 0.25 |
| `ml_drift_features_count` | Gauge | Number of features above alert threshold |
| `ml_prediction_confidence_mean` | Gauge | Rolling mean prediction probability |
| `ml_prediction_confidence_std` | Gauge | Rolling std of prediction probabilities |
| `ml_prediction_default_rate` | Gauge | Rolling default rate in live predictions |
| `ml_predictions_total{decision}` | Counter | Total predictions by decision |
| `ml_prediction_latency_seconds` | Histogram | End-to-end prediction latency |
| `mlflow_run_metric{experiment,metric}` | Gauge | Latest MLflow run metrics |
| `ml_dataset_feature_mean{dataset,feature}` | Gauge | Feature means by dataset |
| `ml_schema_feature_mean{feature}` | Gauge | Training-set feature means (baseline) |

---

## Alert rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| `ModelDriftDetected` | `ml_drift_detected == 1` for 2m | Warning |
| `PredictionConfidenceLow` | `ml_prediction_confidence_mean < 0.35` for 5m | Warning |
| `ModelAccuracyDegraded` | `mlflow_run_metric{metric="test_roc_auc"} < 0.75` | Critical |
| `HighDefaultRate` | `ml_prediction_default_rate > 0.50` for 10m | Warning |
| `InferenceServerDown` | `ml_inference_server_up == 0` for 1m | Critical |

Runbooks: [`docs/runbooks/`](docs/runbooks/)

---

## Project structure


02-observability-ml-alerting/
![02-observability-ml-alerting](docs/diagrams/project _folder_structure_overview.png)

---

## Related Projects

- [01-self-healing-infrastructure](https://github.com/thomasasamba-bot/01-self-healing-infrastructure) — AIOps self-healing with Lambda/SSM
- [03-secure-aws-infrastructure](https://github.com/thomasasamba-bot/03-secure-aws-infrastructure) — IaC with KMS and IAM hardening
- [04-kubernetes-orchestration](https://github.com/thomasasamba-bot/04-kubernetes-orchestration) — EKS zero-downtime deployments
- [05-devsecops-pipeline](https://github.com/thomasasamba-bot/05-devsecops-pipeline) — CI/CD with SonarQube and Trivy

---

*Built by [Thomas Asamba](https://linkedin.com/in/thomasasamba) | [github.com/thomasasamba-bot](https://github.com/thomasasamba-bot)*
