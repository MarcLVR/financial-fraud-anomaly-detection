"""
Integration tests for the fraud scoring API.

Run with:
    pytest tests/test_api.py -v

The tests use FastAPI's TestClient, which runs the app in-process
without needing a running server. This means the model artefacts
must exist in src/models/ before running the tests.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}

LEGITIMATE_TRANSACTION = {
    "step": 10,
    "type": "PAYMENT",
    "amount": 50.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4950.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 1050.0,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_score_returns_expected_fields():
    response = client.post("/score", json=VALID_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert "xgb_probability" in data
    assert "anomaly_score" in data
    assert "risk_score" in data
    assert "risk_band" in data
    assert "action" in data
    assert "top_features" in data


def test_score_risk_band_values():
    response = client.post("/score", json=VALID_TRANSACTION)
    assert response.status_code == 200
    band = response.json()["risk_band"]
    assert band in {"Low", "Medium", "High", "Critical"}


def test_score_probabilities_in_range():
    response = client.post("/score", json=VALID_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["xgb_probability"] <= 1.0
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 1.0


def test_score_top_features_count():
    response = client.post("/score", json=VALID_TRANSACTION)
    assert response.status_code == 200
    assert len(response.json()["top_features"]) == 5


def test_legitimate_transaction_scores_lower():
    fraud_response = client.post("/score", json=VALID_TRANSACTION)
    legit_response = client.post("/score", json=LEGITIMATE_TRANSACTION)
    assert fraud_response.status_code == 200
    assert legit_response.status_code == 200
    fraud_score = fraud_response.json()["risk_score"]
    legit_score = legit_response.json()["risk_score"]
    assert fraud_score > legit_score


def test_invalid_transaction_missing_field():
    bad_transaction = {k: v for k, v in VALID_TRANSACTION.items() if k != "amount"}
    response = client.post("/score", json=bad_transaction)
    assert response.status_code == 422


def test_invalid_transaction_negative_amount():
    bad_transaction = {**VALID_TRANSACTION, "amount": -100.0}
    response = client.post("/score", json=bad_transaction)
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"fraud_api_requests_total" in response.content
