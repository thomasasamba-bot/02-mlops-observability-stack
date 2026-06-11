# Runbook: ModelDriftDetected

**Alert:** `ModelDriftDetected`
**Severity:** Warning
**Team:** MLOps
**Service:** `inference-server` (port 8006)

---

## What this alert means

The PSI (Population Stability Index) for one or more input features has exceeded
the alert threshold of 0.25. This means the distribution of incoming inference
traffic has shifted significantly from the training data distribution.

**This does not mean predictions are wrong yet** — but it does mean the model is
operating outside the conditions it was trained on, and accuracy degradation is
likely if the drift is sustained.

---

## Immediate actions (first 5 minutes)

**1. Identify which features are drifting**

```bash
curl -s http://localhost:8006/drift/report | python3 -m json.tool
```

Look at the `features` array. Features with `severity: "alert"` and `psi > 0.25`
are the problem. Features with `severity: "warning"` are early signals.

**2. Check confidence degradation**

```bash
curl -s http://localhost:8006/drift/status | python3 -m json.tool
```

If `confidence_mean` is dropping toward 0.5, the model is becoming uncertain —
predictions are unreliable. If `confidence_mean` is still above 0.6, the model
may still be performing acceptably despite the distribution shift.

**3. Check the default rate**

```bash
curl -s http://localhost:8006/status | python3 -m json.tool | grep default_rate
```

A `default_rate` above 0.50 in recent predictions is a red flag — either the
model is mispredicting, or the incoming applicant pool has genuinely deteriorated.

---

## Investigation steps

**4. Quantify the drift magnitude**

```bash
python -m app.pipeline.drift_detector \
  --current-csv data/raw/credit_drifted.csv
```

Compare PSI values against thresholds:
- PSI 0.10–0.25 → moderate, monitor
- PSI > 0.25 → significant, action required
- PSI > 0.50 → severe, retrain immediately

**5. Identify the root cause**

Common causes by feature:

| Feature | Likely cause |
|---------|-------------|
| `income` | Economic conditions, applicant pool change, data pipeline issue |
| `credit_score` | Scoring model change upstream, portfolio mix shift |
| `debt_to_income` | Interest rate changes, economic shock |
| `missed_payments` | Seasonal pattern, collection policy change |

**6. Check for data pipeline issues first**

Before assuming real-world drift, verify the data pipeline:

```bash
# Check if feature values are within expected ranges
curl -s http://localhost:8007/metrics | grep ml_dataset_feature_mean
```

Compare against training schema baselines:
```bash
curl -s http://localhost:8007/metrics | grep ml_schema_feature_mean
```

If the means are wildly different, suspect an upstream data issue.

---

## Resolution paths

### Path A: Data pipeline issue (false alarm)
Fix the upstream data issue. Drift should clear within one detector cycle (120s)
once normal traffic resumes.

### Path B: Real distribution shift, model still acceptable
1. Monitor confidence_mean — if it stays above 0.55, defer retraining
2. Set a calendar reminder to retrain within 7 days
3. Increase drift monitoring frequency: `DRIFT_INTERVAL=60`

### Path C: Real distribution shift, model degraded
1. Trigger retraining immediately:
   ```bash
   python -m app.pipeline.train
   ```
2. Verify the new model metrics in MLflow: `http://localhost:5000`
3. The new model will be loaded automatically within `MODEL_RELOAD_INTERVAL` (300s)
4. Verify drift clears after retraining

---

## Escalation

Escalate to senior MLOps if:
- PSI > 1.0 on more than 3 features simultaneously
- `confidence_mean` < 0.35 for more than 30 minutes
- Retraining does not resolve the drift within 2 cycles

---

## Related alerts
- `PredictionConfidenceLow` — usually fires together with severe drift
- `HighDefaultRate` — downstream consequence of sustained drift
- `ModelAccuracyDegraded` — fires when ROC-AUC drops in MLflow