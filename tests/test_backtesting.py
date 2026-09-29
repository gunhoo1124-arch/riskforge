import numpy as np
import pandas as pd
import pytest

from riskforge.backtesting import (
    cumulative_coverage_frame,
    evaluate_violation_independence,
    rolling_var_backtest,
)


def returns_series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_backtest_uses_non_overlapping_forecast_periods() -> None:
    rng = np.random.default_rng(7)
    returns = returns_series(rng.normal(0.0002, 0.01, size=140))

    result = rolling_var_backtest(
        returns,
        training_window=100,
        horizon=5,
        paths=500,
        confidence=0.95,
        seed=3,
    )

    assert result.observations == 8
    outcome_dates = pd.to_datetime(result.forecasts["outcome_date"])
    assert (outcome_dates.diff().dropna().dt.days >= 5).all()


def test_backtest_is_reproducible() -> None:
    rng = np.random.default_rng(9)
    returns = returns_series(rng.normal(0.0, 0.015, size=150))

    first = rolling_var_backtest(returns, training_window=100, paths=500, seed=22)
    second = rolling_var_backtest(returns, training_window=100, paths=500, seed=22)

    pd.testing.assert_frame_equal(first.forecasts, second.forecasts)
    assert first.kupiec_p_value == pytest.approx(second.kupiec_p_value)
    assert first.independence_p_value == pytest.approx(second.independence_p_value)


def test_backtest_detects_losses_beyond_zero_return_training_sample() -> None:
    returns = returns_series(np.concatenate([np.zeros(100), np.full(10, -0.20)]))

    result = rolling_var_backtest(
        returns,
        training_window=100,
        horizon=1,
        paths=500,
        confidence=0.95,
    )

    assert result.violations >= 1
    assert bool(result.forecasts.iloc[0]["violation"])
    assert result.forecasts.iloc[0]["realized_loss"] == pytest.approx(1 - np.exp(-0.20))


def test_cumulative_coverage_matches_final_summary() -> None:
    rng = np.random.default_rng(11)
    returns = returns_series(rng.normal(0.0, 0.01, size=180))
    result = rolling_var_backtest(returns, training_window=100, paths=500)

    coverage = cumulative_coverage_frame(result)

    assert len(coverage) == result.observations
    assert coverage.iloc[-1]["actual_violations"] == result.violations
    assert coverage.iloc[-1]["expected_violations"] == pytest.approx(result.expected_violations)


def test_backtest_rejects_insufficient_history() -> None:
    returns = returns_series(np.zeros(30))

    with pytest.raises(ValueError, match="Not enough returns"):
        rolling_var_backtest(returns, training_window=30, horizon=5)


def test_independence_test_detects_clustered_violations() -> None:
    violations = np.array([False] * 50 + [True] * 10 + [False] * 50)

    result = evaluate_violation_independence(violations)

    assert result.transition_11 == 9
    assert result.violation_after_violation_rate == pytest.approx(0.9)
    assert result.violation_after_safe_rate == pytest.approx(1 / 99)
    assert result.p_value < 0.05
    assert result.status == "clustered"
    assert result.max_consecutive_violations == 10


def test_independence_test_marks_too_few_violations_as_limited() -> None:
    violations = np.array([False, False, True, False, False])

    result = evaluate_violation_independence(violations)

    assert result.status == "limited_evidence"
    assert result.max_consecutive_violations == 1
    assert result.transition_00 + result.transition_01 + result.transition_10 == 4


def test_conditional_coverage_combines_coverage_and_independence() -> None:
    rng = np.random.default_rng(17)
    returns = returns_series(rng.normal(0.0, 0.01, size=220))

    result = rolling_var_backtest(returns, training_window=100, paths=500)

    assert result.conditional_coverage_statistic == pytest.approx(
        result.kupiec_statistic + result.independence_statistic
    )
    assert 0.0 <= result.conditional_coverage_p_value <= 1.0


def test_backtest_supports_reproducible_moving_blocks() -> None:
    rng = np.random.default_rng(31)
    returns = returns_series(rng.normal(0.0, 0.012, size=180))

    first = rolling_var_backtest(
        returns,
        training_window=100,
        horizon=5,
        paths=500,
        method="moving-block",
        block_size=3,
        seed=14,
    )
    second = rolling_var_backtest(
        returns,
        training_window=100,
        horizon=5,
        paths=500,
        method="moving-block",
        block_size=3,
        seed=14,
    )

    pd.testing.assert_frame_equal(first.forecasts, second.forecasts)
