"""Return transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate continuously compounded returns from positive prices.

    The result is ``log(P_t / P_(t-1))`` and therefore has one fewer observation
    than the input series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be provided as a pandas Series.")
    if len(prices) < 2:
        raise ValueError("At least two prices are required to calculate returns.")
    if prices.isna().any():
        raise ValueError("Prices cannot contain missing values.")

    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Prices must contain only finite values.")
    if (values <= 0).any():
        raise ValueError("Prices must be strictly positive.")

    result = np.log(prices.astype(float) / prices.astype(float).shift(1)).dropna()
    result.name = "log_return"
    return result
