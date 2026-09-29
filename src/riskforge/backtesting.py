"""Walk-forward backtesting for market-risk forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, chi2

from riskforge.simulation import BootstrapMethod, normalize_bootstrap_method, simulate_bootstrap


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Forecast-level results and aggregate VaR backtest diagnostics."""

    forecasts: pd.DataFrame
    confidence: float
    observations: int
    violations: int
    expected_violations: float
    observed_violation_rate: float
    expected_violation_rate: float
    violation_rate_ci_low: float
    violation_rate_ci_high: float
    kupiec_statistic: float
    kupiec_p_value: float
    status: str
    transition_00: int
    transition_01: int
    transition_10: int
    transition_11: int
    violation_after_safe_rate: float | None
    violation_after_violation_rate: float | None
    independence_statistic: float
    independence_p_value: float
    independence_status: str
    conditional_coverage_statistic: float
    conditional_coverage_p_value: float
    max_consecutive_violations: int


@dataclass(frozen=True, slots=True)
class IndependenceResult:
    """Transition counts and Christoffersen independence-test diagnostics."""

    transition_00: int
    transition_01: int
    transition_10: int
    transition_11: int
    violation_after_safe_rate: float | None
    violation_after_violation_rate: float | None
    statistic: float
    p_value: float
    status: str
    max_consecutive_violations: int


def _bernoulli_log_likelihood(successes: int, trials: int, rate: float) -> float:
    """Return a numerically safe Bernoulli log likelihood."""
    failures = trials - successes
    success_term = 0.0 if successes == 0 else successes * np.log(rate)
    failure_term = 0.0 if failures == 0 else failures * np.log1p(-rate)
    return float(success_term + failure_term)


def evaluate_violation_independence(
    violations: pd.Series | np.ndarray | list[bool],
) -> IndependenceResult:
    """Test whether consecutive VaR violations behave independently.

    The Christoffersen likelihood-ratio test compares one common violation rate
    with separate rates after safe and violating periods. A statistically
    significant increase after a violation is classified as clustering.
    """
    values = np.asarray(violations)
    if values.ndim != 1:
        raise ValueError("Violations must be a one-dimensional sequence.")
    if values.dtype.kind not in {"b", "i", "u"}:
        raise TypeError("Violations must contain Boolean or 0/1 values.")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Violations must contain only Boolean or 0/1 values.")

    indicators = values.astype(bool)
    if len(indicators) < 2:
        return IndependenceResult(
            transition_00=0,
            transition_01=0,
            transition_10=0,
            transition_11=0,
            violation_after_safe_rate=None,
            violation_after_violation_rate=None,
            statistic=0.0,
            p_value=1.0,
            status="limited_evidence",
            max_consecutive_violations=int(indicators.sum()),
        )

    previous = indicators[:-1]
    current = indicators[1:]
    transition_00 = int((~previous & ~current).sum())
    transition_01 = int((~previous & current).sum())
    transition_10 = int((previous & ~current).sum())
    transition_11 = int((previous & current).sum())

    after_safe_count = transition_00 + transition_01
    after_violation_count = transition_10 + transition_11
    total_transitions = len(indicators) - 1
    total_next_violations = transition_01 + transition_11

    pooled_rate = total_next_violations / total_transitions
    after_safe_rate = transition_01 / after_safe_count if after_safe_count else None
    after_violation_rate = transition_11 / after_violation_count if after_violation_count else None

    null_log_likelihood = _bernoulli_log_likelihood(
        total_next_violations,
        total_transitions,
        pooled_rate,
    )
    alternative_log_likelihood = 0.0
    if after_safe_count:
        assert after_safe_rate is not None
        alternative_log_likelihood += _bernoulli_log_likelihood(
            transition_01,
            after_safe_count,
            after_safe_rate,
        )
    if after_violation_count:
        assert after_violation_rate is not None
        alternative_log_likelihood += _bernoulli_log_likelihood(
            transition_11,
            after_violation_count,
            after_violation_rate,
        )

    statistic = max(
        0.0,
        -2.0 * (null_log_likelihood - alternative_log_likelihood),
    )
    p_value = float(chi2.sf(statistic, df=1))

    violation_count = int(indicators.sum())
    if violation_count < 3 or after_violation_count < 2:
        status = "limited_evidence"
    elif p_value < 0.05 and (
        after_violation_rate is not None
        and after_safe_rate is not None
        and after_violation_rate > after_safe_rate
    ):
        status = "clustered"
    elif p_value < 0.05:
        status = "dependence_detected"
    else:
        status = "no_clustering_detected"

    longest_streak = 0
    current_streak = 0
    for violation in indicators:
        current_streak = current_streak + 1 if violation else 0
        longest_streak = max(longest_streak, current_streak)

    return IndependenceResult(
        transition_00=transition_00,
        transition_01=transition_01,
        transition_10=transition_10,
        transition_11=transition_11,
        violation_after_safe_rate=after_safe_rate,
        violation_after_violation_rate=after_violation_rate,
        statistic=statistic,
        p_value=p_value,
        status=status,
        max_consecutive_violations=longest_streak,
    )


def _kupiec_coverage_test(
    violations: int, observations: int, expected_rate: float
) -> tuple[float, float]:
    """Return the likelihood-ratio statistic and p-value for unconditional coverage."""
    observed_rate = violations / observations

    def log_likelihood(rate: float) -> float:
        successes = 0.0 if violations == 0 else violations * np.log(rate)
        failures_count = observations - violations
        failures = 0.0 if failures_count == 0 else failures_count * np.log1p(-rate)
        return float(successes + failures)

    null_log_likelihood = log_likelihood(expected_rate)
    if observed_rate == 0:
        alternative_log_likelihood = observations * np.log1p(-observed_rate)
    elif observed_rate == 1:
        alternative_log_likelihood = observations * np.log(observed_rate)
    else:
        alternative_log_likelihood = log_likelihood(observed_rate)

    statistic = max(0.0, -2.0 * (null_log_likelihood - alternative_log_likelihood))
    return statistic, float(chi2.sf(statistic, df=1))


def _coverage_status(
    observations: int,
    violations: int,
    expected_rate: float,
    p_value: float,
) -> str:
    """Classify coverage without overstating what a backtest can establish."""
    expected_violations = observations * expected_rate
    if expected_violations < 5:
        return "limited_evidence"
    observed_rate = violations / observations
    if p_value < 0.05 and observed_rate > expected_rate:
        return "risk_underestimated"
    if p_value < 0.05 and observed_rate < expected_rate:
        return "risk_overestimated"
    return "no_failure_detected"


def summarize_var_backtest(
    forecasts: pd.DataFrame,
    confidence: float,
) -> BacktestResult:
    """Calculate coverage diagnostics from forecast-level VaR results."""
    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("Forecasts must be provided as a pandas DataFrame.")
    required_columns = {
        "forecast_date",
        "outcome_date",
        "predicted_var",
        "predicted_expected_shortfall",
        "realized_return",
        "realized_loss",
        "violation",
    }
    missing_columns = required_columns.difference(forecasts.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Forecasts are missing required columns: {missing}.")
    if forecasts.empty:
        raise ValueError("At least one forecast is required for backtest diagnostics.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between 0 and 1.")

    observations = len(forecasts)
    violations = int(forecasts["violation"].astype(bool).sum())
    expected_rate = 1.0 - confidence
    observed_rate = violations / observations
    interval = binomtest(violations, observations).proportion_ci(
        confidence_level=0.95,
        method="wilson",
    )
    kupiec_statistic, kupiec_p_value = _kupiec_coverage_test(
        violations,
        observations,
        expected_rate,
    )
    independence = evaluate_violation_independence(forecasts["violation"])
    conditional_coverage_statistic = kupiec_statistic + independence.statistic
    conditional_coverage_p_value = float(chi2.sf(conditional_coverage_statistic, df=2))

    return BacktestResult(
        forecasts=forecasts,
        confidence=confidence,
        observations=observations,
        violations=violations,
        expected_violations=observations * expected_rate,
        observed_violation_rate=observed_rate,
        expected_violation_rate=expected_rate,
        violation_rate_ci_low=float(interval.low),
        violation_rate_ci_high=float(interval.high),
        kupiec_statistic=kupiec_statistic,
        kupiec_p_value=kupiec_p_value,
        status=_coverage_status(
            observations,
            violations,
            expected_rate,
            kupiec_p_value,
        ),
        transition_00=independence.transition_00,
        transition_01=independence.transition_01,
        transition_10=independence.transition_10,
        transition_11=independence.transition_11,
        violation_after_safe_rate=independence.violation_after_safe_rate,
        violation_after_violation_rate=independence.violation_after_violation_rate,
        independence_statistic=independence.statistic,
        independence_p_value=independence.p_value,
        independence_status=independence.status,
        conditional_coverage_statistic=conditional_coverage_statistic,
        conditional_coverage_p_value=conditional_coverage_p_value,
        max_consecutive_violations=independence.max_consecutive_violations,
    )


def rolling_var_backtest(
    log_returns: pd.Series,
    training_window: int,
    horizon: int = 1,
    paths: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
    method: BootstrapMethod | str = BootstrapMethod.IID,
    block_size: int = 5,
) -> BacktestResult:
    """Backtest historical-bootstrap VaR with non-overlapping forecast periods.

    For each forecast, the simulator uses only the preceding ``training_window``
    returns. Forecast origins advance by ``horizon`` observations, preventing the
    realized evaluation periods from overlapping. The selected bootstrap method is
    applied independently at every historical checkpoint.
    """
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
        samples = simulate_bootstrap(
            training_returns,
            horizon=horizon,
            paths=paths,
            method=selected_method,
            block_size=block_size,
            seed=forecast_seed,
        )
        simulated_terminal_returns = np.expm1(samples.sum(axis=1))
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


def cumulative_coverage_frame(result: BacktestResult) -> pd.DataFrame:
    """Build cumulative actual, expected, and 95% binomial coverage paths."""
    trials = np.arange(1, result.observations + 1)
    actual = result.forecasts["violation"].astype(int).cumsum().to_numpy()
    expected_rate = result.expected_violation_rate
    return pd.DataFrame(
        {
            "outcome_date": result.forecasts["outcome_date"].to_numpy(),
            "actual_violations": actual,
            "expected_violations": trials * expected_rate,
            "lower_95_bound": binom.ppf(0.025, trials, expected_rate),
            "upper_95_bound": binom.ppf(0.975, trials, expected_rate),
        }
    )
