"""Historical stress analysis that preserves the order of realized returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskforge.metrics import expected_shortfall, value_at_risk


@dataclass(frozen=True, slots=True)
class HistoricalStressResult:
    """Rolling historical outcomes and a summary of severe realized periods."""

    horizon: int
    periods: pd.DataFrame
    worst_episodes: pd.DataFrame
    observations: int
    probability_of_loss: float
    probability_exceeding_threshold: float
    historical_var_95: float
    historical_var_99: float
    historical_expected_shortfall_95: float
    worst_loss: float
    worst_start_date: pd.Timestamp
    worst_end_date: pd.Timestamp


def _distinct_worst_episodes(
    periods: pd.DataFrame,
    horizon: int,
    top_n: int,
) -> pd.DataFrame:
    """Select the worst non-overlapping windows to avoid duplicate crash episodes."""
    selected_positions: list[tuple[int, int]] = []
    selected_rows: list[int] = []

    for row_index in periods.sort_values("loss", ascending=False).index:
        start_position = int(periods.at[row_index, "start_position"])
        end_position = start_position + horizon
        is_distinct = all(
            end_position <= selected_start or start_position >= selected_end
            for selected_start, selected_end in selected_positions
        )
        if is_distinct:
            selected_positions.append((start_position, end_position))
            selected_rows.append(int(row_index))
        if len(selected_rows) == top_n:
            break

    return (
        periods.loc[selected_rows]
        .sort_values("loss", ascending=False)
        .drop(columns="start_position")
        .reset_index(drop=True)
    )


def analyze_historical_stress(
    prices: pd.Series,
    horizon: int,
    loss_threshold: float = 0.10,
    top_n: int = 5,
) -> HistoricalStressResult:
    """Measure actual rolling outcomes over the requested holding period.

    Unlike the Monte Carlo bootstrap, this analysis keeps realized market days in
    their original order. All rolling windows are retained for charts and descriptive
    frequencies; the worst-episode table removes overlapping windows so one selloff is
    not presented repeatedly as several separate events.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be provided as a pandas Series.")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Horizon must be a positive integer number of trading days.")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("Top_n must be a positive integer.")
    if not np.isfinite(loss_threshold) or not 0 <= loss_threshold <= 1:
        raise ValueError("Loss threshold must be between 0 and 1.")
    if len(prices) <= horizon:
        raise ValueError(
            "Not enough prices for the requested historical stress horizon. "
            "Choose a longer history window or shorter holding period."
        )
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must have a unique, chronological index.")

    values = prices.to_numpy(dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must contain only positive finite values.")

    dates = pd.to_datetime(prices.index, errors="coerce")
    if dates.isna().any():
        raise ValueError("Price index must contain valid dates.")

    start_prices = values[:-horizon]
    end_prices = values[horizon:]
    returns = end_prices / start_prices - 1.0
    losses = -returns
    periods = pd.DataFrame(
        {
            "start_date": dates[:-horizon],
            "end_date": dates[horizon:],
            "start_price": start_prices,
            "end_price": end_prices,
            "return": returns,
            "loss": losses,
            "start_position": np.arange(len(returns)),
        }
    )
    worst_row = periods.loc[periods["loss"].idxmax()]

    return HistoricalStressResult(
        horizon=horizon,
        periods=periods.drop(columns="start_position"),
        worst_episodes=_distinct_worst_episodes(periods, horizon, top_n),
        observations=len(periods),
        probability_of_loss=float(np.mean(losses > 0)),
        probability_exceeding_threshold=float(np.mean(losses >= loss_threshold)),
        historical_var_95=value_at_risk(losses, 0.95),
        historical_var_99=value_at_risk(losses, 0.99),
        historical_expected_shortfall_95=expected_shortfall(losses, 0.95),
        worst_loss=float(worst_row["loss"]),
        worst_start_date=pd.Timestamp(worst_row["start_date"]),
        worst_end_date=pd.Timestamp(worst_row["end_date"]),
    )


def historical_exceedance_rate(result: HistoricalStressResult, cutoff: float) -> float:
    """Return the share of historical rolling losses strictly worse than a cutoff."""
    if not np.isfinite(cutoff):
        raise ValueError("Loss cutoff must be finite.")
    return float(np.mean(result.periods["loss"].to_numpy() > cutoff))
