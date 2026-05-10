"""
Feature engineering pipeline.

This module contains the same transformation applied in notebook 02.
It is extracted here so the API can import it directly without
depending on the notebook environment.
"""

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transform raw transaction fields into the engineered feature set.

    Applies the same transformations used during training:
    log scaling, balance residuals, ratio features, and one-hot
    encoding of transaction type. The raw monetary columns are dropped
    after the derived features are computed.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw transaction data. Must contain the columns:
        ``amount``, ``oldbalanceOrg``, ``newbalanceOrig``,
        ``oldbalanceDest``, ``newbalanceDest``, ``type``, ``step``.

    Returns
    -------
    pd.DataFrame
        Engineered features ready for scaling and model input.

    Notes
    -----
    ``pd.get_dummies`` uses ``dtype=int`` to avoid the boolean dtype
    that ``select_dtypes(include=[np.number])`` would silently drop.
    """
    out = frame.copy()

    out["log_amount"] = np.log1p(out["amount"])

    balance_error = out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
    out["abs_balance_error"] = balance_error.abs()
    out["log_balance_error"] = np.log1p(out["abs_balance_error"])

    out["log_amount_balance_error"] = np.log1p(
        out["amount"] * out["abs_balance_error"]
    )

    epsilon = 1e-6
    out["log_amount_to_balance_ratio"] = np.log1p(
        out["amount"] / (out["oldbalanceOrg"] + epsilon)
    )
    out["dest_balance_change"] = out["newbalanceDest"] - out["oldbalanceDest"]

    out = pd.get_dummies(out, columns=["type"], drop_first=True, dtype=int)

    raw_columns_to_drop = [
        "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]
    out = out.drop(columns=raw_columns_to_drop)

    return out
