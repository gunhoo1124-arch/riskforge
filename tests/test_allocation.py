import numpy as np
import pandas as pd
import pytest

from riskforge.allocation import one_way_turnover, optimize_allocation


def _allocation_prices() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    common = rng.normal(0.0002, 0.003, 500)
    log_returns = np.column_stack(
        (
            common + rng.normal(0.0002, 0.002, 500),
            common + rng.normal(0.0003, 0.012, 500),
            -0.2 * common + rng.normal(0.0001, 0.006, 500),
        )
    )
    return pd.DataFrame(
        100 * np.exp(np.vstack((np.zeros(3), np.cumsum(log_returns, axis=0)))),
        index=pd.bdate_range("2024-01-02", periods=501),
        columns=["LOW", "HIGH", "DIVERSE"],
    )


def test_one_way_turnover_is_half_absolute_weight_change() -> None:
    turnover = one_way_turnover([0.6, 0.3, 0.1], [0.4, 0.4, 0.2])

    assert turnover == pytest.approx(0.2)


def test_minimum_volatility_optimizer_respects_constraints() -> None:
    result = optimize_allocation(
        _allocation_prices(),
        [0.4, 0.3, 0.3],
        objective="minimum-volatility",
        horizon=20,
        paths=500,
        seed=8,
        max_weight=0.70,
        turnover_limit=0.30,
        expected_shortfall_limit=0.50,
        drawdown_limit=0.50,
    )

    assert result.optimized_weights.sum() == pytest.approx(1.0)
    assert result.optimized_weights.max() <= 0.70 + 2e-5
    assert result.turnover <= 0.30 + 2e-5
    assert (
        result.comparison.iloc[1]["annualized_volatility"]
        <= result.comparison.iloc[0]["annualized_volatility"]
    )
    assert result.constraint_status["passes"].all()


def test_optimizer_is_deterministic() -> None:
    kwargs = {
        "objective": "minimum-expected-shortfall",
        "horizon": 10,
        "paths": 300,
        "seed": 17,
        "max_weight": 0.75,
        "turnover_limit": 0.25,
    }

    first = optimize_allocation(_allocation_prices(), [0.4, 0.3, 0.3], **kwargs)
    second = optimize_allocation(_allocation_prices(), [0.4, 0.3, 0.3], **kwargs)

    np.testing.assert_allclose(first.optimized_weights, second.optimized_weights)
    pd.testing.assert_frame_equal(first.comparison, second.comparison)


def test_zero_turnover_keeps_current_weights() -> None:
    current = np.array([0.4, 0.3, 0.3])

    result = optimize_allocation(
        _allocation_prices(),
        current,
        objective="minimum-volatility",
        horizon=10,
        paths=300,
        max_weight=0.80,
        turnover_limit=0.0,
    )

    np.testing.assert_allclose(result.optimized_weights, current, atol=2e-5)
    assert result.turnover <= 2e-5


def test_incompatible_position_and_turnover_caps_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires at least"):
        optimize_allocation(
            _allocation_prices(),
            [0.9, 0.05, 0.05],
            horizon=10,
            paths=200,
            max_weight=0.60,
            turnover_limit=0.10,
        )


def test_risk_adjusted_objective_uses_shrunk_returns() -> None:
    result = optimize_allocation(
        _allocation_prices(),
        [0.4, 0.3, 0.3],
        objective="risk-adjusted",
        horizon=10,
        paths=300,
        seed=3,
        max_weight=0.70,
        turnover_limit=0.20,
        return_shrinkage=1.0,
    )

    shrunk = result.expected_return_estimates["shrunk_annualized_return"]
    assert np.allclose(shrunk, shrunk.iloc[0])
    assert np.isfinite(result.comparison["risk_adjusted_score"]).all()
