#!/usr/bin/env bash
# =============================================================================
# scripts/bootstrap/setup.sh
#
# One-time project setup: creates virtualenv, installs dependencies,
# generates data, and verifies the stack can start.
#
# Usage:
#   bash scripts/bootstrap/setup.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warning() { echo -e "${YELLOW}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

info "=================================================="
info "  ML Pipeline Observability Stack — Setup"
info "=================================================="

# ── Python version check ──────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || error "Python not found")
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python: $PYTHON ($PY_VERSION)"
[[ "$PY_VERSION" < "3.12" ]] && error "Python 3.12+ required, found $PY_VERSION"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment at .venv ..."
    $PYTHON -m venv .venv
else
    info "Virtual environment already exists at .venv"
fi

source .venv/bin/activate
info "Activated .venv ($(python --version))"

# ── Dependencies ──────────────────────────────────────────────────────────────
info "Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
info "Dependencies installed ✓"

# ── Env file ─────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    info "Created .env from .env.example — review and update as needed"
else
    info ".env already exists"
fi

# ── Directory structure ───────────────────────────────────────────────────────
info "Creating data directories ..."
mkdir -p data/raw data/processed mlruns

# ── Generate data ─────────────────────────────────────────────────────────────
if [ ! -f "data/raw/credit_baseline.csv" ]; then
    info "Generating synthetic datasets ..."
    python scripts/data/generate_data.py
    info "Datasets generated ✓"
else
    info "Datasets already exist — skipping generation"
fi

# ── Smoke test imports ────────────────────────────────────────────────────────
info "Verifying imports ..."
python -c "
import app.pipeline.train
import app.pipeline.predict
import app.pipeline.drift_detector
import app.serving.app
import app.exporter.metrics_exporter
print('All imports OK ✓')
"

info "=================================================="
info "  Setup complete."
info ""
info "  Next steps:"
info "    1. Start MLflow:          mlflow server --host 0.0.0.0 --port 5000"
info "    2. Train model:           python -m app.pipeline.train"
info "    3. Start inference:       uvicorn app.serving.app:app --port 8006 --workers 1"
info "    4. Start exporter:        uvicorn app.exporter.metrics_exporter:app --port 8007"
info "    5. Run tests:             pytest tests/unit/ -v"
info "    6. Full stack (Docker):   bash scripts/deployment/deploy-local.sh"
info "=================================================="