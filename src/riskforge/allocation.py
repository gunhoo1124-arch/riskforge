"""Constrained educational portfolio-allocation sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from riskforge.portfolio import (
    calculate_portfolio_log_returns,
    simulate_joint_bootstrap,
    validate_weights,
)
from riskforge.simulation import BootstrapMethod, normalize_bootstrap_method

FloatArray = NDArray[np.float64]
TRADING_DAYS = 252


class AllocationObjective(StrEnum):
    """Supported allocation sandbox objectives."""

    MINIMUM_VOLATILITY = "minimum-volatility"
    MINIMUM_ES = "minimum-expected-shortfall"
    RISK_ADJUSTED = "risk-adjusted"


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Current-versus-sandbox allocation comparison."""

    tickers: tuple[str, ...]
    objective: AllocationObjective
    current_weights: FloatArray
    optimized_weights: FloatArray
    weight_changes: pd.DataFrame
    comparison: pd.DataFrame
    constraint_status: pd.DataFrame
    expected_return_estimates: pd.DataFrame
    current_paths: FloatArray
    optimized_paths: FloatArray
    turnover: float
    max_weight: float
    turnover_limit: float
    expected_shortfall_limit: float | None
    drawdown_limit: float | None
    return_shrinkage: float
    horizon: int
    scenario_paths: int
    seed: int
    initial_value: float
    iterations: int


@dataclass(frozen=True, slots=True)
class _AllocationMetrics:
    annualized_expected_return: float
    annualized_volatility: float
    risk_adjusted_score: float
    var_95: float
    expected_shortfall_95: float
    probability_of_loss: float
    maximum_drawdown_95: float
    median_terminal_return: float
    price_paths: FloatArray


def normalize_allocation_objective(
    objective: AllocationObjective | str,
) -> AllocationObjective:
    """Return a validated allocation objective."""
    try:
        return AllocationObjective(objective)
    except ValueError as exc:
        choices = ", ".join(item.value for item in AllocationObjective)
        raise ValueError(f"Unknown allocation objective '{objective}'. Choose: {choices}.") from exc


def one_way_turnover(weights: ArrayLike, reference_weights: ArrayLike) -> float:
    """Return one-way turnover, equal to half the absolute weight changes."""
    values = np.asarray(weights, dtype=float)
    reference = np.asarray(reference_weights, dtype=float)
    if values.shape != reference.shape or values.ndim != 1:
        raise ValueError("Weights and reference weights must be matching one-dimensional arrays.")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("Turnover weights must contain only finite values.")
    return float(0.5 * np.abs(values - reference).sum())


def _project_to_capped_simplex(values: FloatArray, cap: float) -> FloatArray:
    """Project values onto non-negative weights summing to one with an upper cap."""
    asset_count = len(values)
    if cap * asset_count < 1.0 - 1e-12:
        raise ValueError(
            f"A {cap:.1%} position cap is impossible with {asset_count} holdings. "
            f"Use at least {1 / asset_count:.1%}."
        )
    lower = float(np.min(values - cap))
    upper = float(np.max(values))
    for _ in range(100):
        midpoint = (lower + upper) / 2
        projected = np.clip(values - midpoint, 0.0, cap)
        if projected.sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - (lower + upper) / 2, 0.0, cap)
    residual = 1.0 - projected.sum()
    if abs(residual) > 1e-10:
        room = cap - projected if residual > 0 else projected
        index = int(np.argmax(room))
        projected[index] += residual
    return projected


def _path_metrics(
    asset_relatives: FloatArray,
    weights: FloatArray,
    expected_returns: FloatArray,
    annualized_covariance: FloatArray,
) -> _AllocationMetrics:
    portfolio_paths = np.einsum("pha,a->ph", asset_relatives, weights, optimize=True)
    terminal_returns = portfolio_paths[:, -1] - 1.0
    losses = -terminal_returns
    var_95 = float(np.quantile(losses, 0.95))
    expected_shortfall_95 = float(losses[losses >= var_95].mean())
    running_peaks = np.maximum.accumulate(portfolio_paths, axis=1)
    maximum_drawdowns = np.max(1.0 - portfolio_paths / running_peaks, axis=1)
    annualized_expected_return = float(weights @ expected_returns)
    variance = float(weights @ annualized_covariance @ weights)
    annualized_volatility = float(np.sqrt(max(variance, 0.0)))
    risk_adjusted_score = (
        annualized_expected_return / annualized_volatility
        if annualized_volatility > np.finfo(float).eps
        else float("-inf")
    )
    return _AllocationMetrics(
        annualized_expected_return=annualized_expected_return,
        annualized_volatility=annualized_volatility,
        risk_adjusted_score=risk_adjusted_score,
        var_95=var_95,
        expected_shortfall_95=expected_shortfall_95,
        probability_of_loss=float(np.mean(terminal_returns < 0)),
        maximum_drawdown_95=float(np.quantile(maximum_drawdowns, 0.95)),
        median_terminal_return=float(np.median(terminal_returns)),
        price_paths=portfolio_paths,
    )


def optimize_allocation(
    prices: pd.DataFrame,
    current_weights: ArrayLike,
    *,
    objective: AllocationObjective | str = AllocationObjective.MINIMUM_ES,
    horizon: int = 20,
    paths: int = 1_000,
    seed: int = 42,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
    max_weight: float = 0.60,
    turnover_limit: float = 0.30,
    expected_shortfall_limit: float | None = None,
    drawdown_limit: float | None = None,
    return_shrinkage: float = 0.75,
    initial_value: float = 100_000.0,
) -> AllocationResult:
    """Find a constrained long-only scenario allocation.

    All candidate weights are evaluated on one deterministic joint-bootstrap
    scenario cube. This is an educational in-sample sandbox, not a trade engine.
    """
    selected_objective = normalize_allocation_objective(objective)
    returns = calculate_portfolio_log_returns(prices)
    tickers = tuple(str(column).upper() for column in prices.columns)
    current = validate_weights(current_weights, len(tickers))
    if horizon < 1:
        raise ValueError("Horizon must be at least one trading day.")
    if paths < 100:
        raise ValueError("Use at least 100 scenario paths for allocation analysis.")
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Initial portfolio value must be a positive finite number.")
    if not 0 < max_weight <= 1:
        raise ValueError("Maximum holding weight must be between 0% and 100%.")
    if not 0 <= turnover_limit <= 1:
        raise ValueError("Turnover limit must be between 0% and 100%.")
    if not 0 <= return_shrinkage <= 1:
        raise ValueError("Return shrinkage must be between 0% and 100%.")
    for value, label in (
        (expected_shortfall_limit, "Expected Shortfall limit"),
        (drawdown_limit, "drawdown limit"),
    ):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{label} must be between 0% and 100%.")

    capped_current = _project_to_capped_simplex(current, max_weight)
    minimum_required_turnover = one_way_turnover(capped_current, current)
    if minimum_required_turnover > turnover_limit + 1e-8:
        raise ValueError(
            f"The position cap requires at least {minimum_required_turnover:.1%} turnover, "
            f"but the turnover limit is {turnover_limit:.1%}. Relax one of these constraints."
        )

    simulated_returns = simulate_joint_bootstrap(
        returns.to_numpy(),
        horizon=horizon,
        paths=paths,
        method=normalize_bootstrap_method(method),
        block_size=block_size,
        seed=seed,
    )
    future_relatives = np.exp(np.cumsum(simulated_returns, axis=1))
    asset_relatives = np.concatenate(
        (np.ones((paths, 1, len(tickers)), dtype=float), future_relatives),
        axis=1,
    )

    daily_means = returns.mean().to_numpy(dtype=float)
    common_daily_mean = float(daily_means.mean())
    shrunk_daily_means = (
        1.0 - return_shrinkage
    ) * daily_means + return_shrinkage * common_daily_mean
    annualized_expected_returns = np.expm1(shrunk_daily_means * TRADING_DAYS)
    raw_annualized_returns = np.expm1(daily_means * TRADING_DAYS)
    annualized_covariance = returns.cov().to_numpy(dtype=float) * TRADING_DAYS

    metrics_cache: dict[bytes, _AllocationMetrics] = {}

    def metrics_for(weights: FloatArray) -> _AllocationMetrics:
        values = np.asarray(weights, dtype=float)
        cache_key = values.tobytes()
        if cache_key not in metrics_cache:
            metrics_cache[cache_key] = _path_metrics(
                asset_relatives,
                values,
                annualized_expected_returns,
                annualized_covariance,
            )
        return metrics_cache[cache_key]

    def objective_value(weights: FloatArray) -> float:
        metrics = metrics_for(weights)
        if selected_objective is AllocationObjective.MINIMUM_VOLATILITY:
            return metrics.annualized_volatility**2
        if selected_objective is AllocationObjective.MINIMUM_ES:
            return metrics.expected_shortfall_95
        return -metrics.risk_adjusted_score

    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: float(turnover_limit - one_way_turnover(weights, current)),
        },
    ]
    if expected_shortfall_limit is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    expected_shortfall_limit - metrics_for(weights).expected_shortfall_95
                ),
            }
        )
    if drawdown_limit is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    drawdown_limit - metrics_for(weights).maximum_drawdown_95
                ),
            }
        )

    rng = np.random.default_rng(seed + 10_000)
    starts = [capped_current]
    equal_weights = np.full(len(tickers), 1.0 / len(tickers))
    for candidate in [equal_weights, *rng.dirichlet(np.ones(len(tickers)), size=8)]:
        projected = _project_to_capped_simplex(np.asarray(candidate, dtype=float), max_weight)
        if one_way_turnover(projected, current) <= turnover_limit + 1e-10:
            starts.append(projected)

    bounds = [(0.0, max_weight) for _ in tickers]
    best_weights: FloatArray | None = None
    best_objective = float("inf")
    best_iterations = 0

    def is_feasible(weights: FloatArray) -> bool:
        values = np.asarray(weights, dtype=float)
        if abs(values.sum() - 1.0) > 2e-5:
            return False
        if (values < -2e-6).any() or (values > max_weight + 2e-5).any():
            return False
        if one_way_turnover(values, current) > turnover_limit + 2e-5:
            return False
        metrics = metrics_for(values)
        if (
            expected_shortfall_limit is not None
            and metrics.expected_shortfall_95 > expected_shortfall_limit + 2e-5
        ):
            return False
        return not (
            drawdown_limit is not None and metrics.maximum_drawdown_95 > drawdown_limit + 2e-5
        )

    for start in starts:
        optimized = minimize(
            objective_value,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-10, "disp": False},
        )
        candidate_weights = np.asarray(optimized.x, dtype=float)
        if is_feasible(candidate_weights):
            candidate_objective = objective_value(candidate_weights)
            if candidate_objective < best_objective:
                best_weights = candidate_weights.copy()
                best_objective = candidate_objective
                best_iterations = int(optimized.nit)

    if best_weights is None:
        current_metrics = metrics_for(current)
        details = []
        if expected_shortfall_limit is not None:
            details.append(
                f"current ES is {current_metrics.expected_shortfall_95:.1%} versus a "
                f"{expected_shortfall_limit:.1%} cap"
            )
        if drawdown_limit is not None:
            details.append(
                f"current drawdown cutoff is {current_metrics.maximum_drawdown_95:.1%} "
                f"versus a {drawdown_limit:.1%} cap"
            )
        suffix = "; ".join(details)
        raise ValueError(
            "No feasible allocation was found under the selected limits. Relax the position, "
            "turnover, ES, or drawdown constraint. " + suffix
        )

    best_weights = _project_to_capped_simplex(best_weights, max_weight)
    current_metrics = metrics_for(current)
    optimized_metrics = metrics_for(best_weights)
    turnover = one_way_turnover(best_weights, current)

    def comparison_row(label: str, metrics: _AllocationMetrics) -> dict[str, float | str]:
        return {
            "allocation": label,
            "annualized_expected_return": metrics.annualized_expected_return,
            "annualized_volatility": metrics.annualized_volatility,
            "risk_adjusted_score": metrics.risk_adjusted_score,
            "var_95": metrics.var_95,
            "expected_shortfall_95": metrics.expected_shortfall_95,
            "probability_of_loss": metrics.probability_of_loss,
            "maximum_drawdown_95": metrics.maximum_drawdown_95,
            "median_terminal_return": metrics.median_terminal_return,
        }

    weight_changes = pd.DataFrame(
        {
            "ticker": tickers,
            "current_weight": current,
            "sandbox_weight": best_weights,
            "weight_change": best_weights - current,
            "hypothetical_dollar_change": (best_weights - current) * initial_value,
        }
    )
    comparison = pd.DataFrame(
        [
            comparison_row("Current allocation", current_metrics),
            comparison_row("Constrained sandbox", optimized_metrics),
        ]
    )

    status_rows: list[dict[str, float | str | bool]] = [
        {
            "constraint": "Weights total",
            "observed": float(best_weights.sum()),
            "limit": 1.0,
            "passes": bool(np.isclose(best_weights.sum(), 1.0, atol=2e-5)),
        },
        {
            "constraint": "Largest holding",
            "observed": float(best_weights.max()),
            "limit": max_weight,
            "passes": bool(best_weights.max() <= max_weight + 2e-5),
        },
        {
            "constraint": "One-way turnover",
            "observed": turnover,
            "limit": turnover_limit,
            "passes": bool(turnover <= turnover_limit + 2e-5),
        },
    ]
    if expected_shortfall_limit is not None:
        status_rows.append(
            {
                "constraint": "95% Expected Shortfall",
                "observed": optimized_metrics.expected_shortfall_95,
                "limit": expected_shortfall_limit,
                "passes": bool(
                    optimized_metrics.expected_shortfall_95 <= expected_shortfall_limit + 2e-5
                ),
            }
        )
    if drawdown_limit is not None:
        status_rows.append(
            {
                "constraint": "95% maximum drawdown",
                "observed": optimized_metrics.maximum_drawdown_95,
                "limit": drawdown_limit,
                "passes": bool(optimized_metrics.maximum_drawdown_95 <= drawdown_limit + 2e-5),
            }
        )

    expected_return_estimates = pd.DataFrame(
        {
            "ticker": tickers,
            "raw_historical_annualized_return": raw_annualized_returns,
            "shrunk_annualized_return": annualized_expected_returns,
        }
    )
    return AllocationResult(
        tickers=tickers,
        objective=selected_objective,
        current_weights=current,
        optimized_weights=best_weights,
        weight_changes=weight_changes,
        comparison=comparison,
        constraint_status=pd.DataFrame(status_rows),
        expected_return_estimates=expected_return_estimates,
        current_paths=current_metrics.price_paths,
        optimized_paths=optimized_metrics.price_paths,
        turnover=turnover,
        max_weight=max_weight,
        turnover_limit=turnover_limit,
        expected_shortfall_limit=expected_shortfall_limit,
        drawdown_limit=drawdown_limit,
        return_shrinkage=return_shrinkage,
        horizon=horizon,
        scenario_paths=paths,
        seed=seed,
        initial_value=float(initial_value),
        iterations=best_iterations,
    )
