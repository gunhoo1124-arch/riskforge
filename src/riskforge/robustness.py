"""Systematic bootstrap-model and lookback sensitivity comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskforge.backtesting import rolling_var_backtest
from riskforge.metrics import summarize_risk
from riskforge.returns import calculate_log_returns
from riskforge.simulation import (
    BootstrapMethod,
    normalize_bootstrap_method,
    returns_to_price_paths,
    simulate_bootstrap,
)


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """A distinct historical-bootstrap configuration."""

    label: str
    method: BootstrapMethod
    block_size: int


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    """Side-by-side model, backtest, and lookback comparison results."""

    model_comparison: pd.DataFrame
    backtest_comparison: pd.DataFrame
    lookback_comparison: pd.DataFrame
    horizon: int
    paths: int
    seed: int
    training_window: int
    evaluation_returns: int


def model_configurations(
    horizon: int,
    block_sizes: tuple[int, ...] = (3, 5, 10),
) -> tuple[ModelConfiguration, ...]:
    """Build unique model configurations for a forecast horizon."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")
    if not block_sizes:
        raise ValueError("At least one moving-block size is required.")

    configurations = [
        ModelConfiguration(
            label="Independent days (IID)",
            method=BootstrapMethod.IID,
            block_size=1,
        )
    ]
    seen_effective_sizes: set[int] = set()
    for requested_size in block_sizes:
        if (
            not isinstance(requested_size, int)
            or isinstance(requested_size, bool)
            or requested_size < 1
        ):
            raise ValueError("Block sizes must be positive integers.")
        effective_size = min(requested_size, horizon)
        if effective_size == 1 or effective_size in seen_effective_sizes:
            continue
        seen_effective_sizes.add(effective_size)
        configurations.append(
            ModelConfiguration(
                label=f"Moving blocks: {effective_size} days",
                method=BootstrapMethod.MOVING_BLOCK,
                block_size=effective_size,
            )
        )
    return tuple(configurations)


def _risk_row(
    label: str,
    method: BootstrapMethod,
    block_size: int,
    historical_returns: np.ndarray,
    initial_price: float,
    horizon: int,
    paths: int,
    seed: int,
    loss_threshold: float,
) -> dict[str, object]:
    """Simulate one configuration and return comparable risk statistics."""
    simulated_returns = simulate_bootstrap(
        historical_returns,
        horizon=horizon,
        paths=paths,
        method=method,
        block_size=block_size,
        seed=seed,
    )
    price_paths = returns_to_price_paths(simulated_returns, initial_price)
    summary = summarize_risk(price_paths, loss_threshold)
    return {
        "model": label,
        "method": method.value,
        "block_size": block_size if method is BootstrapMethod.MOVING_BLOCK else None,
        "var_95": summary.var_95,
        "var_99": summary.var_99,
        "expected_shortfall_95": summary.expected_shortfall_95,
        "probability_of_loss": summary.probability_of_loss,
        "probability_exceeding_threshold": summary.probability_exceeding_threshold,
        "maximum_drawdown_95": summary.maximum_drawdown_95,
    }


def _available_lookbacks(available_returns: int) -> tuple[int, ...]:
    """Return useful nested windows, always including the full available sample."""
    candidates = (252, 504, 756, 1_260, 2_520)
    windows = {window for window in candidates if window <= available_returns}
    windows.add(available_returns)
    return tuple(sorted(windows))


def _lookback_label(window: int, full_window: int) -> str:
    """Format a trading-day lookback without implying exact calendar duration."""
    if window == full_window:
        return f"Full sample ({window:,} days)"
    years = window / 252
    return f"About {years:g} years ({window:,} days)"


def analyze_model_robustness(
    prices: pd.Series,
    horizon: int = 20,
    paths: int = 1_000,
    seed: int = 42,
    loss_threshold: float = 0.10,
    selected_method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    selected_block_size: int = 5,
    block_sizes: tuple[int, ...] = (3, 5, 10),
) -> RobustnessResult:
    """Compare bootstrap assumptions, backtests, and historical lookback windows."""
    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be provided as a pandas Series.")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")
    if paths < 100:
        raise ValueError("At least 100 paths are required for robustness analysis.")
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    if not 0 <= loss_threshold <= 1:
        raise ValueError("Loss threshold must be between 0 and 1.")
    if (
        not isinstance(selected_block_size, int)
        or isinstance(selected_block_size, bool)
        or selected_block_size < 1
    ):
        raise ValueError("Selected block size must be a positive integer.")

    selected = normalize_bootstrap_method(selected_method)
    log_returns = calculate_log_returns(prices)
    values = log_returns.to_numpy(dtype=float)
    if len(values) < max(100, horizon * 2 + 20):
        raise ValueError(
            "Not enough history for a useful robustness comparison at this horizon. "
            "Choose a shorter horizon or longer history window."
        )
    if selected_block_size > len(values):
        raise ValueError("Selected block size cannot exceed the available history.")

    comparison_sizes = tuple(sorted({*block_sizes, selected_block_size}))
    configurations = model_configurations(horizon, comparison_sizes)
    initial_price = float(prices.iloc[-1])
    model_rows = [
        _risk_row(
            configuration.label,
            configuration.method,
            configuration.block_size,
            values,
            initial_price,
            horizon,
            paths,
            seed,
            loss_threshold,
        )
        for configuration in configurations
    ]

    training_window = min(756, max(60, len(values) // 2))
    backtest_rows: list[dict[str, object]] = []
    for configuration in configurations:
        backtest = rolling_var_backtest(
            log_returns,
            training_window=training_window,
            horizon=horizon,
            paths=paths,
            confidence=0.95,
            seed=seed,
            method=configuration.method,
            block_size=configuration.block_size,
        )
        backtest_rows.append(
            {
                "model": configuration.label,
                "observations": backtest.observations,
                "violations": backtest.violations,
                "expected_violations": backtest.expected_violations,
                "violation_rate": backtest.observed_violation_rate,
                "kupiec_p_value": backtest.kupiec_p_value,
                "independence_p_value": backtest.independence_p_value,
                "conditional_coverage_p_value": backtest.conditional_coverage_p_value,
                "coverage_status": backtest.status,
                "independence_status": backtest.independence_status,
                "longest_violation_run": backtest.max_consecutive_violations,
            }
        )

    selected_effective_size = min(selected_block_size, horizon)
    lookback_rows: list[dict[str, object]] = []
    for window in _available_lookbacks(len(values)):
        risk_row = _risk_row(
            _lookback_label(window, len(values)),
            selected,
            selected_effective_size,
            values[-window:],
            initial_price,
            horizon,
            paths,
            seed,
            loss_threshold,
        )
        risk_row["lookback"] = window
        lookback_rows.append(risk_row)

    return RobustnessResult(
        model_comparison=pd.DataFrame(model_rows),
        backtest_comparison=pd.DataFrame(backtest_rows),
        lookback_comparison=pd.DataFrame(lookback_rows),
        horizon=horizon,
        paths=paths,
        seed=seed,
        training_window=training_window,
        evaluation_returns=len(values) - training_window,
    )
