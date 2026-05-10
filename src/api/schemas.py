"""
Request and response schemas for the scoring API.

Pydantic v2 validates every incoming request against the
TransactionRequest model before the scoring pipeline runs.
Invalid requests are rejected with a 422 error before they
touch any model code.
"""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """A single raw transaction as received from the client."""

    step: int = Field(..., description="Time step in the simulation (integer).")
    type: str = Field(
        ...,
        description=(
            "Transaction type. One of: PAYMENT, TRANSFER, CASH_OUT, "
            "CASH_IN, DEBIT."
        ),
    )
    amount: float = Field(..., gt=0, description="Transaction amount in local currency.")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before transaction.")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after transaction.")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before transaction.")
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after transaction.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 181.0,
                    "oldbalanceOrg": 181.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                }
            ]
        }
    }


class FeatureContribution(BaseModel):
    """SHAP contribution of a single feature to the fraud score."""

    feature: str
    value: float
    shap_contribution: float


class ScoreResponse(BaseModel):
    """Scoring result returned for a single transaction."""

    xgb_probability: float = Field(
        ..., description="Calibrated fraud probability from XGBoost (0 to 1)."
    )
    anomaly_score: float = Field(
        ..., description="Normalised Isolation Forest anomaly score (0 to 1)."
    )
    risk_score: float = Field(
        ..., description="Combined risk score: 0.7 × XGBoost + 0.3 × anomaly."
    )
    risk_band: str = Field(
        ..., description="Risk band: Low, Medium, High, or Critical."
    )
    action: str = Field(
        ..., description="Recommended action for the risk band."
    )
    top_features: list[FeatureContribution] = Field(
        ..., description="Top 5 SHAP feature contributions driving the score."
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: bool
