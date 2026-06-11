# Runbook: InferenceServerDown

**Alert:** `InferenceServerDown`
**Severity:** Critical
**Team:** MLOps
**Service:** `inference-server` (port 8006)

---

## What this alert means

The inference server's `/health/live` endpoint has been returning non-200
or timing out for 1 minute. No predictions are being served.

---

## Immediate actions (first 2 minutes)

**1. Check if the process is running**

```bash
# Local / Docker Compose
curl -sf http://localhost:8006/health/live

# Kubernetes
kubectl get pods -n mlops -l app=inference-server
kubectl describe pod -n mlops -l app=inference-server
```

**2. Check the logs**

```bash
# Docker Compose
docker compose logs inference-server --tail=50

# Kubernetes
kubectl logs -n mlops -l app=inference-server --tail=50
```

**3. Check if MLflow is reachable**

The inference server loads the model from MLflow on startup. If MLflow is down,
the server will hang on readiness probe.

```bash
curl -sf http://localhost:5000/health
# or
kubectl get pods -n mlops -l app=mlflow
```

---

## Common causes and fixes

### Cause 1: MLflow server unreachable at startup
The inference server failed to load the model because MLflow was down when it
started. The server is alive but not ready.

```bash
# Check readiness separately
curl -sf http://localhost:8006/health/ready
```

**Fix:** Restart the inference server after MLflow is healthy.
```bash
# Docker Compose
docker compose restart inference-server

# Kubernetes
kubectl rollout restart deployment/inference-server -n mlops
```

### Cause 2: No model in MLflow registry
The server started but couldn't find a `credit-scoring-model` in Staging stage.

```bash
# Check MLflow registry
curl -s http://localhost:5000/api/2.0/mlflow/registered-models/get \
  ?name=credit-scoring-model | python3 -m json.tool
```

**Fix:** Run training to register a model.
```bash
python -m app.pipeline.train
# Then restart inference server
```

### Cause 3: OOMKilled (Kubernetes)
```bash
kubectl describe pod -n mlops -l app=inference-server | grep -A5 "OOMKilled"
```

**Fix:** Increase memory limit in `infra/kubernetes/deployments/inference-deployment.yaml`
and reapply.

### Cause 4: Port conflict (local)
```bash
lsof -i :8006
```

**Fix:** Kill the conflicting process or change the port.

---

## Restart procedure

```bash
# Local
uvicorn app.serving.app:app --host 0.0.0.0 --port 8006 --workers 1

# Docker Compose
docker compose restart inference-server

# Kubernetes
kubectl rollout restart deployment/inference-server -n mlops
kubectl rollout status deployment/inference-server -n mlops --timeout=120s
```

---

## Verify recovery

```bash
curl -s http://localhost:8006/health/live  | python3 -m json.tool
curl -s http://localhost:8006/health/ready | python3 -m json.tool
curl -s -X POST http://localhost:8006/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"income":60000,"loan_amount":15000,"credit_score":680,
       "debt_to_income":0.35,"employment_years":5,"num_credit_lines":4,
       "missed_payments":0}' | python3 -m json.tool
```

All three should succeed. The predict call should return `"decision": "NO_DEFAULT"`.

---

## Escalation

Escalate if the server does not recover within 10 minutes after restart,
or if the model cannot be loaded after MLflow is confirmed healthy.