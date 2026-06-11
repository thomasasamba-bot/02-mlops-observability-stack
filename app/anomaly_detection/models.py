from pydantic import BaseModel
from typing import Dict
from datetime import datetime


class MetricSample(BaseModel):
    metric_name: str
    value: float
    timestamp: datetime


class AnomalyResult(BaseModel):
    metric_name: str
    anomaly_score: float
    detection_method: str
    timestamp: datetime
    metadata: Dict = {}