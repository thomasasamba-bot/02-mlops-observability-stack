# =============================================================================
# infra/docker/base-images/python-ml-base.Dockerfile
#
# Base image for all Python ML services in this project.
# Used by: inference server, metrics exporter, training job.
#
# Build:
#   docker build -f infra/docker/base-images/python-ml-base.Dockerfile \
#     -t observability-ml:latest .
#
# The image is also what the Kubernetes manifests reference as
# `image: observability-ml:latest`.
# =============================================================================

FROM python:3.12-slim

# Build args
ARG BUILD_DATE
ARG GIT_SHA
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.title="observability-ml" \
      org.opencontainers.image.description="ML Pipeline Observability Stack"

# System deps — minimal, only what's needed for scipy/sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      g++ \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mluser

WORKDIR /app

# Install Python deps first (layer-cached as long as requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/         ./app/
COPY scripts/     ./scripts/
COPY tests/       ./tests/
COPY data/        ./data/

# Create directories the app writes to at runtime
RUN mkdir -p data/raw data/processed mlruns && \
    chown -R mluser:mluser /app

USER mluser

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Default: inference server
# Override with `docker run ... uvicorn app.exporter.metrics_exporter:app ...`
# or `docker run ... python -m app.pipeline.train`
EXPOSE 8006
CMD ["uvicorn", "app.serving.app:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "1"]