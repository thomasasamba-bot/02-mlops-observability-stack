#!/usr/bin/env bash
# =============================================================================
# scripts/deployment/destroy.sh
#
# Tear down the local Docker Compose stack OR Kubernetes deployment.
#
# Usage:
#   bash scripts/deployment/destroy.sh           # stop Docker Compose stack
#   bash scripts/deployment/destroy.sh --k8s     # delete K8s namespace (all resources)
#   bash scripts/deployment/destroy.sh --all     # both
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[destroy]${NC} $*"; }
warn()  { echo -e "${RED}[destroy]${NC} $*"; }

K8S=false
for arg in "$@"; do
    [[ "$arg" == "--k8s" || "$arg" == "--all" ]] && K8S=true
done

DO_COMPOSE=true
[[ "$1" == "--k8s" ]] && DO_COMPOSE=false

# ── Docker Compose ────────────────────────────────────────────────────────────
if [ "$DO_COMPOSE" = true ]; then
    COMPOSE="docker compose"
    command -v docker-compose >/dev/null 2>&1 && COMPOSE="docker-compose"
    if $COMPOSE ps -q 2>/dev/null | grep -q .; then
        info "Stopping Docker Compose stack ..."
        $COMPOSE down --volumes --remove-orphans
        info "Docker Compose stack stopped ✓"
    else
        info "No Docker Compose stack running"
    fi
fi

# ── Kubernetes ────────────────────────────────────────────────────────────────
if [ "$K8S" = true ]; then
    warn "This will DELETE the mlops namespace and ALL resources in it."
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Deleting mlops namespace ..."
        kubectl delete namespace mlops --ignore-not-found=true
        kubectl delete clusterrole    prometheus-mlops --ignore-not-found=true
        kubectl delete clusterrolebinding prometheus-mlops --ignore-not-found=true
        info "Kubernetes resources deleted ✓"
    else
        info "Kubernetes deletion cancelled"
    fi
fi

info "Done."