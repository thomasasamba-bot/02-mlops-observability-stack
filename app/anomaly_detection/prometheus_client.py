import requests
from .config import PROMETHEUS_URL


def query_prometheus(query: str):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10
    )

    response.raise_for_status()

    return response.json()