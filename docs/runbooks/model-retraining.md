# Runbook: Model Retraining

**Trigger:** Scheduled (weekly), or on `ModelDriftDetected` / `ModelAccuracyDegraded` alert
**Team:** MLOps
**Time required:** ~5 minutes

---

## When to retrain

| Trigger | Action |
|---------|--------|
| `ModelDriftDetected` — PSI > 0.25 on 2+ features | Retrain within 24h |
| `ModelAccuracyDegraded` — ROC-AUC < 0.75 in MLflow | Retrain immediately |
| Weekly scheduled run | Retrain with latest data |
| Significant upstream data schema change | Retrain + update feature schema |

---

## Pre-retraining checklist

- [ ] MLflow server is running and accessible
- [ ] New training data is available (or existing baseline is still valid)
- [ ] Disk space available for new model artefacts

---

## Retraining procedure

**1. (Optional) Regenerate training data**

If the baseline distribution has genuinely shifted and you want to adapt:
```bash
python scripts/data/generate_data.py --samples 10000
```

For production: replace `credit_baseline.csv` with fresh labelled data
from your feature store / data warehouse.

**2. Train the model**

```bash
python -m app.pipeline.train \
  --experiment-name credit-scoring \
  --run-name "retrain-$(date +%Y%m%d)" \
  --n-estimators 150
```

Watch for:
- `CV roc_auc` should be ≥ 0.80
- `TEST roc_auc` should be ≥ 0.78
- Feature importances should be dominated by `credit_score` and `income`

**3. Verify the new model in MLflow**

```bash
open http://localhost:5000
```

Compare the new run against the previous run:
- ROC-AUC should not regress by more than 0.02
- F1 should be ≥ 0.65

**4. The inference server hot-reloads automatically**

The model cache checks for a newer Staging version every `MODEL_RELOAD_INTERVAL`
seconds (default: 300s). You can force an immediate reload by restarting:

```bash
# Docker Compose
docker compose restart inference-server

# Kubernetes
kubectl rollout restart deployment/inference-server -n mlops
```

**5. Verify the new model is serving**

```bash
curl -s http://localhost:8006/status | python3 -m json.tool | grep model_version
```

The `model_version` should match the latest version in MLflow.

**6. Run the drift check to confirm baseline reset**

```bash
python -m app.pipeline.drift_detector \
  --current-csv data/raw/credit_baseline.csv
```

Should show no drift detected.

---

## Rollback procedure

If the new model performs worse in production:

```bash
# Find the previous good version in MLflow
# Then update MODEL_STAGE env var or set a @champion alias
export MODEL_STAGE=Production  # or pin to specific version
docker compose restart inference-server
```

In Kubernetes:
```bash
kubectl set env deployment/inference-server MODEL_STAGE=Production -n mlops
kubectl rollout restart deployment/inference-server -n mlops
```