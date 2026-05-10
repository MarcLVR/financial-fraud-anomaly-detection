"""
Fraud Risk Scoring API.

Exposes three endpoints:
- POST /score   — scores a single transaction and returns a risk band,
                  combined score, and top-5 SHAP feature contributions.
- GET  /health  — confirms the service is up and models are loaded.
- GET  /metrics — Prometheus metrics: request count, latency, errors.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Docker:
    docker build -t fraud-scoring-api .
    docker run -p 8000:8000 fraud-scoring-api
"""

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.api.schemas import HealthResponse, ScoreResponse, TransactionRequest
from src.api.scorer import score_transaction

# Structured JSON logging so log ingestion tools (Splunk, Datadog,
# ELK) can parse fields without regex.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Risk Scoring API",
    description=(
        "Behavioural risk scoring for financial transactions. "
        "Combines a calibrated XGBoost classifier with an Isolation Forest "
        "anomaly detector and returns a risk band with SHAP explanations."
    ),
    version="1.0.0",
)

# Prometheus metrics — three core signals any production ML service
# should expose: request volume, latency distribution, error count.
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total number of scoring requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
ERROR_COUNT = Counter(
    "fraud_api_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record latency and request count for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUEST_COUNT.labels(
        endpoint=endpoint, status=response.status_code
    ).inc()

    return response


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    """Liveness check. Returns 200 if the service is up and models are loaded."""
    return {"status": "ok", "models_loaded": True}


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score(transaction: TransactionRequest):
    """Score a single transaction and return a risk band with explanations.

    The pipeline runs:
    1. Feature engineering (build_features)
    2. RobustScaler transformation
    3. Calibrated XGBoost → fraud probability
    4. Isolation Forest → normalised anomaly score
    5. Combined score = 0.7 × XGBoost + 0.3 × anomaly
    6. Risk band assignment (Low / Medium / High / Critical)
    7. SHAP top-5 feature contributions
    """
    try:
        result = score_transaction(transaction.model_dump())
        logger.info(
            "Scored transaction | band=%s score=%.4f",
            result["risk_band"],
            result["risk_score"],
        )
        return result
    except Exception as exc:
        ERROR_COUNT.labels(endpoint="/score", error_type=type(exc).__name__).inc()
        logger.error("Scoring error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", tags=["Operations"])
def metrics():
    """Prometheus metrics endpoint.

    Exposes request count, latency histogram, and error count.
    Scrape this endpoint with a Prometheus instance or use
    Grafana for dashboards.
    """
    return PlainTextResponse(
        generate_latest(), media_type=CONTENT_TYPE_LATEST
    )
