FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd -m appuser

COPY app/serving/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "app.anomaly_detection.detector:app", "--host", "0.0.0.0", "--port", "8000"]