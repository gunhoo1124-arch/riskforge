"""Transparent rolling trend-and-volatility market-regime analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from riskforge.metrics import expected_shortfall, value_at_risk


class MarketRegime(StrEnum):
    """Human-readable combinations of relative volatility and rolling trend."""

    CALM_ADVANCE = "Calm advance"
    CALM_DECLINE = "Calm decline"
    TURBULENT_ADVANCE = "Turbulent advance"
    TURBULENT_DECLINE = "Turbulent decline"


REGIME_ORDER = (
    MarketRegime.CALM_ADVANCE,
    MarketRegime.CALM_DECLINE,
    MarketRegime.TURBULENT_ADVANCE,
    MarketRegime.TURBULENT_DECLINE,
)


@dataclass(frozen=True, slots=True)
class RegimeAnalysis:
    """Historical regime classifications and subsequent-outcome diagnostics."""

    window: int
    forward_horizon: int
    history: pd.DataFrame
    summary: pd.DataFrame
    volatility_threshold: float
    current_regime: MarketRegime
    current_annualized_volatility: float
    current_rolling_return: float
    outcome_observations: int


def _validate_prices(prices: pd.Series, window: int, forward_horizon: int) -> np.ndarray:
    """Validate regime-analysis inputs and return numeric prices."""
    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be provided as a pandas Series.")
    if not isinstance(window, int) or isinstance(window, bool) or window < 20:
        raise ValueError("Regime window must be an integer of at least 20 trading days.")
    if (
        not isinstance(forward_horizon, int)
        or isinstance(forward_horizon, bool)
        or forward_horizon < 1
    ):
        raise ValueError("Forward horizon must be a positive integer.")
    if len(prices) <= window + forward_horizon:
        raise ValueError(
            "Not enough prices for the regime window and forward horizon. "
            "Choose a longer history or shorter regime outlook."
        )
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must have a unique, chronological index.")

    values = prices.to_numpy(dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must contain only positive finite values.")
    dates = pd.to_datetime(prices.index, errors="coerce")
    if dates.isna().any():
        raise ValueError("Price index must contain valid dates.")
    return values


def _regime_labels(
    rolling_return: pd.Series,
    annualized_volatility: pd.Series,
    volatility_threshold: pd.Series,
) -> np.ndarray:
    """Classify observations from rolling trend and relative volatility."""
    turbulent = annualized_volatility > volatility_threshold
    advancing = rolling_return >= 0
    return np.select(
        [
            ~turbulent & advancing,
            ~turbulent & ~advancing,
            turbulent & advancing,
            turbulent & ~advancing,
        ],
        [regime.value for regime in REGIME_ORDER],
        default=MarketRegime.CALM_ADVANCE.value,
    )


def analyze_market_regimes(
    prices: pd.Series,
    window: int = 63,
    forward_horizon: int = 20,
) -> RegimeAnalysis:
    """Classify historical regimes and summarize what followed each one.

    Volatility is the rolling standard deviation of daily log returns annualized
    with the 252-trading-day convention. "Turbulent" means above the expanding
    median rolling volatility available at that date; advance versus decline is
    determined by the rolling simple return. Forward outcomes are descriptive and
    may overlap.
    """
    _validate_prices(prices, window, forward_horizon)
    indexed_prices = prices.astype(float).copy()
    indexed_prices.index = pd.to_datetime(indexed_prices.index)

    log_returns = np.log(indexed_prices / indexed_prices.shift(1))
    annualized_volatility = log_returns.rolling(window).std(ddof=1) * np.sqrt(252)
    rolling_return = indexed_prices / indexed_prices.shift(window) - 1.0
    forward_return = indexed_prices.shift(-forward_horizon) / indexed_prices - 1.0

    metrics = pd.DataFrame(
        {
            "price": indexed_prices,
            "rolling_return": rolling_return,
            "annualized_volatility": annualized_volatility,
            "forward_return": forward_return,
        }
    ).dropna(subset=["rolling_return", "annualized_volatility"])
    metrics["volatility_threshold"] = metrics["annualized_volatility"].expanding().median()
    metrics["regime"] = _regime_labels(
        metrics["rolling_return"],
        metrics["annualized_volatility"],
        metrics["volatility_threshold"],
    )
    metrics.index.name = "date"
    history = metrics.reset_index()

    outcomes = history.dropna(subset=["forward_return"])
    summary_rows: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        regime_outcomes = outcomes[outcomes["regime"] == regime.value]
        if regime_outcomes.empty:
            continue
        returns = regime_outcomes["forward_return"].to_numpy(dtype=float)
        losses = -returns
        summary_rows.append(
            {
                "regime": regime.value,
                "observations": len(returns),
                "sample_share": len(returns) / len(outcomes),
                "median_forward_return": float(np.median(returns)),
                "probability_of_loss": float(np.mean(returns < 0)),
                "var_95": value_at_risk(losses, 0.95),
                "expected_shortfall_95": expected_shortfall(losses, 0.95),
                "worst_loss": float(losses.max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    current = history.iloc[-1]

    return RegimeAnalysis(
        window=window,
        forward_horizon=forward_horizon,
        history=history,
        summary=summary,
        volatility_threshold=float(current["volatility_threshold"]),
        current_regime=MarketRegime(str(current["regime"])),
        current_annualized_volatility=float(current["annualized_volatility"]),
        current_rolling_return=float(current["rolling_return"]),
        outcome_observations=len(outcomes),
    )
