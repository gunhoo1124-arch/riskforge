"""Multi-asset historical-bootstrap portfolio risk analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from riskforge.metrics import RiskSummary, summarize_risk
from riskforge.quality import DataQualityReport
from riskforge.simulation import BootstrapMethod, normalize_bootstrap_method

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PortfolioRiskResult:
    """Outputs from a buy-and-hold multi-asset risk simulation."""

    tickers: tuple[str, ...]
    weights: FloatArray
    historical_prices: pd.DataFrame
    historical_log_returns: pd.DataFrame
    portfolio_paths: FloatArray
    summary: RiskSummary
    correlation: pd.DataFrame
    asset_risk: pd.DataFrame
    tail_contributions: pd.DataFrame
    weighted_standalone_var_95: float
    weighted_standalone_es_95: float
    var_diversification_gap: float
    es_diversification_gap: float
    effective_number_of_assets: float
    initial_value: float
    loss_threshold: float
    horizon: int
    paths: int
    method: BootstrapMethod
    block_size: int
    seed: int
    data_quality: DataQualityReport | None = None


def validate_weights(weights: ArrayLike, asset_count: int) -> FloatArray:
    """Return validated long-only portfolio weights that sum to one."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size != asset_count:
        raise ValueError("Provide exactly one weight for each portfolio ticker.")
    if not np.isfinite(values).all():
        raise ValueError("Portfolio weights must contain only finite numbers.")
    if (values < 0).any():
        raise ValueError("Portfolio weights cannot be negative in this long-only model.")
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights must add up to exactly 100%.")
    if np.count_nonzero(values) < 2:
        raise ValueError("At least two portfolio assets must have a positive weight.")
    return values.astype(float, copy=True)


def investments_to_weights(amounts: ArrayLike, asset_count: int) -> tuple[FloatArray, float]:
    """Convert positive per-holding dollar amounts into portfolio weights and total value."""
    values = np.asarray(amounts, dtype=float)
    if values.ndim != 1 or values.size != asset_count:
        raise ValueError("Provide exactly one investment amount for each portfolio ticker.")
    if not np.isfinite(values).all():
        raise ValueError("Every holding needs a finite dollar amount.")
    if (values <= 0).any():
        raise ValueError("Every holding must have an investment amount greater than zero.")
    total = float(values.sum())
    if not np.isfinite(total) or total < 100:
        raise ValueError("The portfolio cart total must be at least $100.")
    if total > 1_000_000_000:
        raise ValueError("The portfolio cart total cannot exceed $1 billion.")
    return validate_weights(values / total, asset_count), total


def calculate_portfolio_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate aligned daily log returns for a multi-asset price frame."""
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2 or len(prices) < 3:
        raise ValueError("Prices must contain at least three rows for two or more assets.")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Portfolio prices must contain only finite values.")
    if (numeric <= 0).any().any():
        raise ValueError("Portfolio prices must be strictly positive.")

    returns = np.log(numeric / numeric.shift(1)).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Calculated portfolio returns contain invalid values.")
    return returns


def _validated_return_matrix(historical_returns: ArrayLike) -> FloatArray:
    values = np.asarray(historical_returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(
            "Historical portfolio returns must have shape (at least 2 days, at least 2 assets)."
        )
    if not np.isfinite(values).all():
        raise ValueError("Historical portfolio returns must contain only finite values.")
    return values


def simulate_joint_bootstrap(
    historical_returns: ArrayLike,
    horizon: int = 20,
    paths: int = 10_000,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
    seed: int = 42,
) -> FloatArray:
    """Resample complete multi-asset return rows with a deterministic seed.

    The output has shape ``(paths, horizon, assets)``. Resampling a whole row
    preserves the same-day cross-sectional relationship between assets. Moving
    blocks also retain consecutive historical rows inside each sampled block.
    """
    values = _validated_return_matrix(historical_returns)
    if horizon < 1:
        raise ValueError("Horizon must be at least 1 trading day.")
    if paths < 1:
        raise ValueError("Paths must be at least 1.")
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("Block size must be a positive integer.")
    if block_size > values.shape[0]:
        raise ValueError("Block size cannot exceed the number of historical return rows.")

    selected_method = normalize_bootstrap_method(method)
    rng = np.random.default_rng(seed)
    if selected_method is BootstrapMethod.IID or block_size == 1:
        indexes = rng.integers(0, values.shape[0], size=(paths, horizon))
        return values[indexes]

    effective_block_size = min(block_size, horizon)
    result = np.empty((paths, horizon, values.shape[1]), dtype=float)
    position = 0
    while position < horizon:
        remaining = horizon - position
        current_size = min(effective_block_size, remaining)
        starts = rng.integers(
            0,
            values.shape[0] - effective_block_size + 1,
            size=paths,
        )
        indexes = starts[:, np.newaxis] + np.arange(current_size)
        result[:, position : position + current_size, :] = values[indexes]
        position += current_size
    return result


def buy_and_hold_portfolio_paths(
    simulated_log_returns: ArrayLike,
    weights: ArrayLike,
    initial_value: float = 100_000.0,
) -> tuple[FloatArray, FloatArray]:
    """Convert asset return scenarios into buy-and-hold portfolio value paths.

    Returns the total portfolio paths and asset value relatives. Portfolio weights
    are applied only at day zero and then drift with each asset's simulated value.
    """
    returns = np.asarray(simulated_log_returns, dtype=float)
    if returns.ndim != 3 or min(returns.shape) < 1:
        raise ValueError("Simulated returns must have shape (paths, horizon, assets).")
    if not np.isfinite(returns).all():
        raise ValueError("Simulated returns must contain only finite values.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Initial portfolio value must be a positive finite number.")
    validated_weights = validate_weights(weights, returns.shape[2])

    future_relatives = np.exp(np.cumsum(returns, axis=1))
    initial_relatives = np.ones((returns.shape[0], 1, returns.shape[2]), dtype=float)
    asset_relatives = np.concatenate((initial_relatives, future_relatives), axis=1)
    asset_values = initial_value * validated_weights[np.newaxis, np.newaxis, :] * asset_relatives
    return asset_values.sum(axis=2), asset_relatives


def analyze_portfolio_risk(
    prices: pd.DataFrame,
    weights: ArrayLike,
    *,
    horizon: int = 20,
    paths: int = 10_000,
    initial_value: float = 100_000.0,
    loss_threshold: float = 0.10,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
    seed: int = 42,
) -> PortfolioRiskResult:
    """Run a joint bootstrap and summarize buy-and-hold portfolio downside risk."""
    if not 0 <= loss_threshold <= 1:
        raise ValueError("Loss threshold must be between 0 and 1.")
    tickers = tuple(str(column).upper() for column in prices.columns)
    validated_weights = validate_weights(weights, len(tickers))
    historical_returns = calculate_portfolio_log_returns(prices)
    selected_method = normalize_bootstrap_method(method)
    simulated_returns = simulate_joint_bootstrap(
        historical_returns.to_numpy(),
        horizon=horizon,
        paths=paths,
        method=selected_method,
        block_size=block_size,
        seed=seed,
    )
    portfolio_paths, asset_relatives = buy_and_hold_portfolio_paths(
        simulated_returns,
        validated_weights,
        initial_value,
    )
    summary = summarize_risk(portfolio_paths, loss_threshold)

    asset_rows: list[dict[str, float | str]] = []
    standalone_vars: list[float] = []
    standalone_es: list[float] = []
    annualized_volatility = historical_returns.std(ddof=1) * np.sqrt(252)
    for index, ticker in enumerate(tickers):
        asset_summary = summarize_risk(asset_relatives[:, :, index], loss_threshold)
        standalone_vars.append(asset_summary.var_95)
        standalone_es.append(asset_summary.expected_shortfall_95)
        asset_rows.append(
            {
                "ticker": ticker,
                "weight": validated_weights[index],
                "starting_value": initial_value * validated_weights[index],
                "historical_volatility": float(annualized_volatility.iloc[index]),
                "standalone_var_95": asset_summary.var_95,
                "standalone_es_95": asset_summary.expected_shortfall_95,
            }
        )
    asset_risk = pd.DataFrame(asset_rows)

    initial_allocations = initial_value * validated_weights
    terminal_asset_values = initial_allocations[np.newaxis, :] * asset_relatives[:, -1, :]
    asset_loss_fractions = (
        initial_allocations[np.newaxis, :] - terminal_asset_values
    ) / initial_value
    portfolio_loss_fractions = asset_loss_fractions.sum(axis=1)
    tail_mask = portfolio_loss_fractions >= summary.var_95
    contribution_fractions = asset_loss_fractions[tail_mask].mean(axis=0)
    if np.isclose(summary.expected_shortfall_95, 0.0):
        contribution_shares = np.full(len(tickers), np.nan)
    else:
        contribution_shares = contribution_fractions / summary.expected_shortfall_95
    tail_contributions = pd.DataFrame(
        {
            "ticker": tickers,
            "weight": validated_weights,
            "es_contribution": contribution_fractions,
            "es_contribution_dollars": contribution_fractions * initial_value,
            "share_of_es": contribution_shares,
        }
    )

    weighted_standalone_var = float(np.dot(validated_weights, standalone_vars))
    weighted_standalone_es = float(np.dot(validated_weights, standalone_es))
    return PortfolioRiskResult(
        tickers=tickers,
        weights=validated_weights,
        historical_prices=prices.copy(),
        historical_log_returns=historical_returns,
        portfolio_paths=portfolio_paths,
        summary=summary,
        correlation=historical_returns.corr(),
        asset_risk=asset_risk,
        tail_contributions=tail_contributions,
        weighted_standalone_var_95=weighted_standalone_var,
        weighted_standalone_es_95=weighted_standalone_es,
        var_diversification_gap=weighted_standalone_var - summary.var_95,
        es_diversification_gap=weighted_standalone_es - summary.expected_shortfall_95,
        effective_number_of_assets=float(1.0 / np.sum(validated_weights**2)),
        initial_value=float(initial_value),
        loss_threshold=float(loss_threshold),
        horizon=horizon,
        paths=paths,
        method=selected_method,
        block_size=block_size,
        seed=seed,
    )
