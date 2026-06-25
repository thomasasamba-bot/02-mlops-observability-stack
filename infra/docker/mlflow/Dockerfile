FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.13.0 && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /mlflow-data/artifacts && \
    chmod 755 /mlflow-data && \
    chmod 755 /mlflow-data/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "1", \
     "--allowed-hosts", "127.0.0.1,localhost,mlflow,mlflow:5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/artifacts", \
     "--serve-artifacts"]
