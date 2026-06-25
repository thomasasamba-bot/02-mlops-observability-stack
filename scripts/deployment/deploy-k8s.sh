#!/usr/bin/env bash
# =============================================================================
# scripts/deployment/deploy-k8s.sh
#
# Deploy the full ML observability stack to Kubernetes (minikube or cluster).
#
# Usage:
#   bash scripts/deployment/deploy-k8s.sh           # deploy all
#   bash scripts/deployment/deploy-k8s.sh --build   # build + push image first
#   bash scripts/deployment/deploy-k8s.sh --train   # deploy + run training job
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[deploy-k8s]${NC} $*"; }
warning() { echo -e "${YELLOW}[deploy-k8s]${NC} $*"; }

BUILD=false
TRAIN=false
for arg in "$@"; do
    case $arg in
        --build) BUILD=true ;;
        --train) TRAIN=true ;;
    esac
done

# ── Pre-flight ────────────────────────────────────────────────────────────────
command -v kubectl >/dev/null 2>&1 || { echo "kubectl not found"; exit 1; }
kubectl cluster-info --request-timeout=5s >/dev/null 2>&1 || \
    { echo "No Kubernetes cluster reachable — start minikube first"; exit 1; }

# ── Build image ───────────────────────────────────────────────────────────────
if [ "$BUILD" = true ]; then
    info "Building Docker image ..."
    # For minikube: build directly into minikube's Docker daemon
    if command -v minikube >/dev/null 2>&1; then
        eval "$(minikube docker-env)"
        info "Building into minikube Docker daemon ..."
    fi
    docker build \
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --build-arg GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
        -f infra/docker/base-images/python-ml-base.Dockerfile \
        -t observability-ml:latest .
    info "Image built ✓"
    info "MLflow image built ✓"
fi

# ── Deploy ────────────────────────────────────────────────────────────────────
info "Applying manifests ..."

kubectl apply -f infra/kubernetes/namespace.yaml

# Monitoring (Prometheus needs RBAC before pods start)
kubectl apply -f infra/kubernetes/monitoring/prometheus-deployment.yaml
kubectl apply -f infra/kubernetes/monitoring/alertmanager-deployment.yaml
kubectl apply -f infra/kubernetes/monitoring/grafana-deployment.yaml

# Application services
kubectl apply -f infra/kubernetes/deployments/mlflow-deployment.yaml
kubectl apply -f infra/kubernetes/deployments/inference-deployment.yaml
kubectl apply -f infra/kubernetes/deployments/exporter-deployment.yaml

# Ingress
kubectl apply -f infra/kubernetes/ingress/ingress.yaml

info "Waiting for MLflow / monitoring (no model dependency) ..."
kubectl rollout status deployment/mlflow     -n mlops --timeout=120s
kubectl rollout status deployment/prometheus -n mlops --timeout=120s
kubectl rollout status deployment/grafana    -n mlops --timeout=120s

# NOTE: inference-server's and metrics-exporter's readiness probes
# (/health/ready) return 503 until a model has been registered in MLflow.
# Their `kubectl rollout status` checks are intentionally deferred until
# AFTER the training job runs below — checking them here would time out
# and abort the script (set -e) on every first-time deploy, before training
# ever had a chance to run.

# ── Training job ──────────────────────────────────────────────────────────────
if [ "$TRAIN" = true ]; then
    info "Running training job ..."
    # Delete previous job if exists
    kubectl delete job model-training -n mlops --ignore-not-found=true
    kubectl apply -f infra/kubernetes/jobs/training-job.yaml
    info "Waiting for training job to complete ..."
    kubectl wait --for=condition=complete job/model-training -n mlops --timeout=300s
    info "Training job complete ✓"
else
    warning "Skipping training (--train not set)."
    warning "inference-server / metrics-exporter will report NOT READY"
    warning "(/health/ready -> 503) until a model is registered. Run:"
    warning "  bash scripts/deployment/deploy-k8s.sh --train"
fi

info "Waiting for inference-server / metrics-exporter ..."
if [ "$TRAIN" = true ]; then
    kubectl rollout status deployment/inference-server -n mlops --timeout=120s
    kubectl rollout status deployment/metrics-exporter  -n mlops --timeout=120s
else
    info "(skipped — no model registered yet, see warning above)"
fi

# ── Port-forward helper ───────────────────────────────────────────────────────
info "=================================================="
info "  Deployment complete. Access services with:"
info ""
info "  kubectl port-forward svc/mlflow           5000:5000 -n mlops &"
info "  kubectl port-forward svc/inference-server 8006:8006 -n mlops &"
info "  kubectl port-forward svc/metrics-exporter 8007:8007 -n mlops &"
info "  kubectl port-forward svc/prometheus        9090:9090 -n mlops &"
info "  kubectl port-forward svc/grafana           3000:3000 -n mlops &"
info ""
info "  Or with minikube:  minikube service --all -n mlops"
info "=================================================="