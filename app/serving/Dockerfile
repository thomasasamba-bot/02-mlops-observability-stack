# =============================================================================
# app/serving/Dockerfile
#
# ML inference server image — app.serving.app:app on port 8006.
#
# Build:
#   docker build -f app/serving/Dockerfile -t ml-inference-server:latest .
# =============================================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# curl is required by the HEALTHCHECK below — not present in python:slim by default
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

COPY app/serving/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# app.serving.app imports app.pipeline.* — copy the whole app/ tree, not just serving/
COPY app ./app

# Writable dirs the app needs at runtime (mlruns for local MLflow fallback,
# data/processed for the feature schema written by train.py)
RUN mkdir -p data/raw data/processed mlruns \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8006

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8006/health/live || exit 1

CMD ["uvicorn", "app.serving.app:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "1"]
