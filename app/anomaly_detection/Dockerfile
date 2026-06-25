# =============================================================================
# app/anomaly_detection/Dockerfile
#
# AIOps infrastructure anomaly detection service —
# app.anomaly_detection.detector:app on port 8000.
#
# This previously did not exist: docker-compose.yml's `anomaly-detector`
# service was incorrectly building from app/serving/Dockerfile (the ML
# inference server image), which neither exposed port 8000 nor installed
# this module's dependencies (elasticsearch, etc.).
#
# Build:
#   docker build -f app/anomaly_detection/Dockerfile -t anomaly-detector:latest .
# =============================================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# curl is required by docker-compose's HEALTHCHECK for this service
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

COPY app/anomaly_detection/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Only this module is needed at runtime
COPY app/anomaly_detection ./app/anomaly_detection
COPY app/__init__.py ./app/__init__.py

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "app.anomaly_detection.detector:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
