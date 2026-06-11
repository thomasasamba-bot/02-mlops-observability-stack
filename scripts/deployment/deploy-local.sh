#!/usr/bin/env bash
# =============================================================================
# scripts/deployment/deploy-local.sh
#
# Start the full ML observability stack locally using Docker Compose.
# Runs: MLflow, inference server, metrics exporter, Prometheus,
#       Grafana, Alertmanager, Elasticsearch, Kibana, Logstash.
#
# Usage:
#   bash scripts/deployment/deploy-local.sh           # start all services
#   bash scripts/deployment/deploy-local.sh --build   # rebuild images first
#   bash scripts/deployment/deploy-local.sh --train   # start + train model
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[deploy-local]${NC} $*"; }
warning() { echo -e "${YELLOW}[deploy-local]${NC} $*"; }

BUILD=false
TRAIN=false
for arg in "$@"; do
    case $arg in
        --build) BUILD=true ;;
        --train) TRAIN=true ;;
    esac
done

# ── Pre-flight ────────────────────────────────────────────────────────────────
command -v docker      >/dev/null 2>&1 || { echo "Docker not found"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || \
    docker compose version >/dev/null 2>&1 || { echo "Docker Compose not found"; exit 1; }

COMPOSE="docker compose"
command -v docker-compose >/dev/null 2>&1 && COMPOSE="docker-compose"

[ -f ".env" ] || cp .env.example .env

# ── Generate data if needed ───────────────────────────────────────────────────
if [ ! -f "data/raw/credit_baseline.csv" ]; then
    info "Generating datasets ..."
    source .venv/bin/activate 2>/dev/null || true
    python scripts/data/generate_data.py
fi

# ── Build ─────────────────────────────────────────────────────────────────────
if [ "$BUILD" = true ]; then
    info "Building Docker images ..."
    $COMPOSE build
fi

# ── Start ─────────────────────────────────────────────────────────────────────
info "Starting stack ..."
$COMPOSE up -d

info "Waiting for services to be healthy ..."
sleep 10

# ── Train ─────────────────────────────────────────────────────────────────────
if [ "$TRAIN" = true ]; then
    info "Waiting for MLflow to be ready ..."
    until curl -sf http://localhost:5000/health > /dev/null 2>&1; do
        sleep 3
    done
    info "Training model ..."
    $COMPOSE exec -T inference-server python -m app.pipeline.train || \
        docker run --rm --network host \
            -e MLFLOW_TRACKING_URI=http://localhost:5000 \
            -v "$PWD":/app -w /app \
            observability-ml:latest python -m app.pipeline.train
fi

# ── Status ────────────────────────────────────────────────────────────────────
info "=================================================="
info "  Stack is running:"
info "    MLflow:          http://localhost:5000"
info "    Inference API:   http://localhost:8006/docs"
info "    Metrics export:  http://localhost:8007/metrics"
info "    Prometheus:      http://localhost:9090"
info "    Grafana:         http://localhost:3000  (admin/admin)"
info "    Alertmanager:    http://localhost:9093"
info "=================================================="
info "  Stop with: bash scripts/deployment/destroy.sh"