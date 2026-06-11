#!/usr/bin/env bash
# =============================================================================
# scripts/bootstrap/init-env.sh
#
# Activate the virtual environment and export all .env variables
# into the current shell session.
#
# Usage (source, not execute):
#   source scripts/bootstrap/init-env.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[init-env] Activated .venv"
else
    echo "[init-env] WARNING: .venv not found — run: bash scripts/bootstrap/setup.sh"
fi

# Export .env variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "[init-env] Loaded .env"
else
    echo "[init-env] WARNING: .env not found — copying from .env.example"
    cp .env.example .env
    set -a; source .env; set +a
fi

echo "[init-env] Environment ready. MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"