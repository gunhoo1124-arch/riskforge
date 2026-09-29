"""EWMA-filtered historical simulation and walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from riskforge.backtesting import (
    BacktestResult,
    rolling_var_backtest,
    summarize_var_backtest,
)
from riskforge.metrics import RiskSummary, summarize_risk
from riskforge.returns import calculate_log_returns
from riskforge.simulation import (
    BootstrapMethod,
    normalize_bootstrap_method,
    returns_to_price_paths,
    simulate_bootstrap,
)

FloatArray = NDArray[np.float64]
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class EwmaFilterResult:
    """Historical volatility filter and standardized empirical shocks."""

    conditional_volatility: FloatArray
    standardized_shocks: FloatArray
    drift: float
    next_volatility: float
    decay: float


@dataclass(frozen=True, slots=True)
class EwmaSimulationResult:
    """Simulated log returns plus the volatility state used to create them."""

    simulated_returns: FloatArray
    filter_result: EwmaFilterResult


@dataclass(frozen=True, slots=True)
class VolatilityAnalysisResult:
    """Current and out-of-sample comparison of fixed and adaptive volatility models."""

    current_comparison: pd.DataFrame
    baseline_backtest: BacktestResult
    adaptive_backtest: BacktestResult
    volatility_history: pd.DataFrame
    adaptive_price_paths: FloatArray
    current_annualized_volatility: float
    median_annualized_volatility: float
    decay: float
    horizon: int
    paths: int
    training_window: int
    evaluation_returns: int


def _validated_returns(historical_returns: ArrayLike) -> FloatArray:
    """Return a finite one-dimensional array suitable for EWMA filtering."""
    values = np.asarray(historical_returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("Historical returns must be a one-dimensional array.")
    if values.size < 20:
        raise ValueError("At least 20 historical returns are required for EWMA filtering.")
    if not np.isfinite(values).all():
        raise ValueError("Historical returns must contain only finite values.")
    return values


def _validated_decay(decay: float) -> float:
    """Validate an EWMA decay factor."""
    if isinstance(decay, bool) or not isinstance(decay, (int, float, np.integer, np.floating)):
        raise TypeError("EWMA decay must be a number.")
    normalized = float(decay)
    if not np.isfinite(normalized) or not 0 < normalized < 1:
        raise ValueError("EWMA decay must be strictly between 0 and 1.")
    return normalized


def ewma_filter(
    historical_returns: ArrayLike,
    decay: float = 0.94,
    initialization_window: int = 20,
) -> EwmaFilterResult:
    """Filter returns into time-varying volatility and empirical shocks.

    After initialization, volatility for observation ``t`` uses squared
    innovations only through ``t - 1``. The drift and initial variance are
    estimated from the supplied training sample; an out-of-sample forecast must
    therefore pass only information available at its forecast checkpoint.
    """
    values = _validated_returns(historical_returns)
    selected_decay = _validated_decay(decay)
    if (
        not isinstance(initialization_window, int)
        or isinstance(initialization_window, bool)
        or initialization_window < 2
    ):
        raise ValueError("Initialization window must be an integer of at least 2.")

    drift = float(values.mean())
    innovations = values - drift
    initial_size = min(initialization_window, len(values))
    variance_floor = np.finfo(float).eps
    variance = max(float(np.var(innovations[:initial_size], ddof=1)), variance_floor)
    conditional_volatility = np.empty(len(values), dtype=float)
    conditional_volatility[0] = np.sqrt(variance)

    for index in range(1, len(values)):
        variance = selected_decay * variance + (1 - selected_decay) * innovations[index - 1] ** 2
        variance = max(variance, variance_floor)
        conditional_volatility[index] = np.sqrt(variance)

    next_variance = selected_decay * variance + (1 - selected_decay) * innovations[-1] ** 2
    next_volatility = float(np.sqrt(max(next_variance, variance_floor)))
    standardized_shocks = innovations / conditional_volatility
    return EwmaFilterResult(
        conditional_volatility=conditional_volatility,
        standardized_shocks=standardized_shocks,
        drift=drift,
        next_volatility=next_volatility,
        decay=selected_decay,
    )


def simulate_ewma_filtered_bootstrap(
    historical_returns: ArrayLike,
    horizon: int = 20,
    paths: int = 10_000,
    decay: float = 0.94,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
    seed: int = 42,
    starting_volatility: float | None = None,
) -> EwmaSimulationResult:
    """Simulate empirical shocks while recursively updating EWMA volatility."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")
    if not isinstance(paths, int) or isinstance(paths, bool) or paths < 1:
        raise ValueError("Paths must be a positive integer.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    selected_method = normalize_bootstrap_method(method)
    filtered = ewma_filter(historical_returns, decay=decay)
    if starting_volatility is None:
        initial_volatility = filtered.next_volatility
    else:
        if (
            isinstance(starting_volatility, bool)
            or not isinstance(
                starting_volatility,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(starting_volatility)
            or starting_volatility <= 0
        ):
            raise ValueError("Starting volatility must be a positive finite number.")
        initial_volatility = float(starting_volatility)
    sampled_shocks = simulate_bootstrap(
        filtered.standardized_shocks,
        horizon=horizon,
        paths=paths,
        method=selected_method,
        block_size=block_size,
        seed=seed,
    )

    simulated_returns = np.empty((paths, horizon), dtype=float)
    variance = np.full(paths, initial_volatility**2, dtype=float)
    for day in range(horizon):
        innovations = np.sqrt(variance) * sampled_shocks[:, day]
        simulated_returns[:, day] = filtered.drift + innovations
        variance = filtered.decay * variance + (1 - filtered.decay) * innovations**2

    return EwmaSimulationResult(
        simulated_returns=simulated_returns,
        filter_result=filtered,
    )


def rolling_ewma_var_backtest(
    log_returns: pd.Series,
    training_window: int,
    horizon: int = 1,
    paths: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
    decay: float = 0.94,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
) -> BacktestResult:
    """Backtest EWMA-filtered VaR using only pre-forecast information."""
    if not isinstance(log_returns, pd.Series):
        raise TypeError("Log returns must be provided as a pandas Series.")
    values = log_returns.to_numpy(dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Log returns must be a finite one-dimensional series.")
    if training_window < 20:
        raise ValueError("Training window must contain at least 20 daily returns.")
    if horizon < 1:
        raise ValueError("Horizon must be at least 1 trading day.")
    if paths < 100:
        raise ValueError("At least 100 simulated paths are required for backtesting.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between 0 and 1.")
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    selected_decay = _validated_decay(decay)
    selected_method = normalize_bootstrap_method(method)
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("Block size must be a positive integer.")
    if selected_method is BootstrapMethod.MOVING_BLOCK and block_size > training_window:
        raise ValueError("Block size cannot exceed the training window.")
    if len(values) < training_window + horizon:
        raise ValueError(
            "Not enough returns for the requested training window and forecast horizon."
        )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    final_origin = len(values) - horizon
    for origin in range(training_window, final_origin + 1, horizon):
        training_returns = values[origin - training_window : origin]
        forecast_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        simulation = simulate_ewma_filtered_bootstrap(
            training_returns,
            horizon=horizon,
            paths=paths,
            decay=selected_decay,
            method=selected_method,
            block_size=block_size,
            seed=forecast_seed,
        )
        simulated_terminal_returns = np.expm1(simulation.simulated_returns.sum(axis=1))
        simulated_losses = -simulated_terminal_returns
        var = float(np.quantile(simulated_losses, confidence))
        expected_shortfall = float(simulated_losses[simulated_losses >= var].mean())

        realized_return = float(np.expm1(values[origin : origin + horizon].sum()))
        realized_loss = -realized_return
        rows.append(
            {
                "forecast_date": log_returns.index[origin - 1],
                "outcome_date": log_returns.index[origin + horizon - 1],
                "predicted_var": var,
                "predicted_expected_shortfall": expected_shortfall,
                "realized_return": realized_return,
                "realized_loss": realized_loss,
                "violation": bool(realized_loss > var),
            }
        )

    return summarize_var_backtest(pd.DataFrame(rows), confidence)


def _comparison_row(label: str, summary: RiskSummary) -> dict[str, object]:
    """Convert a risk summary into a display-ready comparison row."""
    return {
        "model": label,
        "var_95": summary.var_95,
        "var_99": summary.var_99,
        "expected_shortfall_95": summary.expected_shortfall_95,
        "expected_shortfall_99": summary.expected_shortfall_99,
        "probability_of_loss": summary.probability_of_loss,
        "probability_exceeding_threshold": summary.probability_exceeding_threshold,
        "maximum_drawdown_95": summary.maximum_drawdown_95,
    }


def analyze_volatility_model(
    prices: pd.Series,
    *,
    horizon: int = 20,
    paths: int = 1_000,
    training_window: int = 252,
    confidence: float = 0.95,
    seed: int = 42,
    decay: float = 0.94,
    loss_threshold: float = 0.10,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
) -> VolatilityAnalysisResult:
    """Compare fixed and EWMA-filtered bootstraps now and out of sample."""
    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be provided as a pandas Series.")
    if paths < 100:
        raise ValueError("At least 100 paths are required for volatility analysis.")
    if not 0 <= loss_threshold <= 1:
        raise ValueError("Loss threshold must be between 0 and 1.")

    log_returns = calculate_log_returns(prices)
    values = log_returns.to_numpy(dtype=float)
    selected_method = normalize_bootstrap_method(method)
    baseline_returns = simulate_bootstrap(
        values,
        horizon=horizon,
        paths=paths,
        method=selected_method,
        block_size=block_size,
        seed=seed,
    )
    adaptive_simulation = simulate_ewma_filtered_bootstrap(
        values,
        horizon=horizon,
        paths=paths,
        decay=decay,
        method=selected_method,
        block_size=block_size,
        seed=seed,
    )

    initial_price = float(prices.iloc[-1])
    baseline_paths = returns_to_price_paths(baseline_returns, initial_price)
    adaptive_paths = returns_to_price_paths(adaptive_simulation.simulated_returns, initial_price)
    baseline_summary = summarize_risk(baseline_paths, loss_threshold)
    adaptive_summary = summarize_risk(adaptive_paths, loss_threshold)

    baseline_backtest = rolling_var_backtest(
        log_returns,
        training_window=training_window,
        horizon=horizon,
        paths=paths,
        confidence=confidence,
        seed=seed,
        method=selected_method,
        block_size=block_size,
    )
    adaptive_backtest = rolling_ewma_var_backtest(
        log_returns,
        training_window=training_window,
        horizon=horizon,
        paths=paths,
        confidence=confidence,
        seed=seed,
        decay=decay,
        method=selected_method,
        block_size=block_size,
    )

    filtered = adaptive_simulation.filter_result
    annualized_history = filtered.conditional_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    volatility_history = pd.DataFrame(
        {
            "date": log_returns.index,
            "annualized_volatility": annualized_history,
        }
    )
    return VolatilityAnalysisResult(
        current_comparison=pd.DataFrame(
            [
                _comparison_row("Fixed historical bootstrap", baseline_summary),
                _comparison_row("EWMA volatility-aware", adaptive_summary),
            ]
        ),
        baseline_backtest=baseline_backtest,
        adaptive_backtest=adaptive_backtest,
        volatility_history=volatility_history,
        adaptive_price_paths=adaptive_paths,
        current_annualized_volatility=(filtered.next_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)),
        median_annualized_volatility=float(np.median(annualized_history)),
        decay=filtered.decay,
        horizon=horizon,
        paths=paths,
        training_window=training_window,
        evaluation_returns=len(values) - training_window,
    )
