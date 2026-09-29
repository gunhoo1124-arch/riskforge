import numpy as np
import pandas as pd
import pytest

from riskforge.volatility import (
    analyze_volatility_model,
    ewma_filter,
    rolling_ewma_var_backtest,
    simulate_ewma_filtered_bootstrap,
)


def sample_returns(size: int = 320, seed: int = 18) -> pd.Series:
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0003, 0.008, size // 2)
    turbulent = rng.normal(-0.0001, 0.022, size - size // 2)
    return pd.Series(
        np.concatenate([calm, turbulent]),
        index=pd.bdate_range("2024-01-02", periods=size),
    )


def sample_prices(size: int = 321) -> pd.Series:
    returns = sample_returns(size - 1)
    values = 100 * np.exp(np.r_[0.0, np.cumsum(returns.to_numpy())])
    return pd.Series(values, index=pd.bdate_range("2024-01-02", periods=size))


def test_ewma_filter_reacts_to_recent_volatility() -> None:
    calm = np.full(100, 0.002)
    volatile = calm.copy()
    volatile[-5:] = np.array([0.08, -0.07, 0.09, -0.06, 0.07])

    calm_result = ewma_filter(calm)
    volatile_result = ewma_filter(volatile)

    assert volatile_result.next_volatility > calm_result.next_volatility
    assert volatile_result.standardized_shocks.shape == (100,)


def test_filtered_simulation_is_reproducible() -> None:
    returns = sample_returns().to_numpy()

    first = simulate_ewma_filtered_bootstrap(returns, horizon=20, paths=250, seed=7)
    second = simulate_ewma_filtered_bootstrap(returns, horizon=20, paths=250, seed=7)

    assert first.simulated_returns.shape == (250, 20)
    np.testing.assert_array_equal(first.simulated_returns, second.simulated_returns)
    assert np.isfinite(first.simulated_returns).all()


def test_filtered_simulation_accepts_external_starting_volatility() -> None:
    returns = sample_returns().to_numpy()
    low = simulate_ewma_filtered_bootstrap(
        returns,
        horizon=5,
        paths=500,
        seed=8,
        starting_volatility=0.005,
    )
    high = simulate_ewma_filtered_bootstrap(
        returns,
        horizon=5,
        paths=500,
        seed=8,
        starting_volatility=0.05,
    )

    assert high.simulated_returns[:, 0].std() > low.simulated_returns[:, 0].std() * 5


def test_walk_forward_backtest_uses_non_overlapping_periods() -> None:
    result = rolling_ewma_var_backtest(
        sample_returns(),
        training_window=160,
        horizon=5,
        paths=200,
        seed=9,
    )

    outcome_positions = sample_returns().index.get_indexer(result.forecasts["outcome_date"])
    assert result.observations == 32
    assert (np.diff(outcome_positions) == 5).all()


def test_first_forecast_does_not_use_later_outcomes() -> None:
    original = sample_returns()
    changed = original.copy()
    changed.iloc[105:] = changed.iloc[105:] * 20

    first = rolling_ewma_var_backtest(
        original,
        training_window=100,
        horizon=5,
        paths=200,
        seed=4,
    )
    second = rolling_ewma_var_backtest(
        changed,
        training_window=100,
        horizon=5,
        paths=200,
        seed=4,
    )

    assert first.forecasts.iloc[0]["predicted_var"] == pytest.approx(
        second.forecasts.iloc[0]["predicted_var"]
    )
    assert first.forecasts.iloc[0]["realized_loss"] == pytest.approx(
        second.forecasts.iloc[0]["realized_loss"]
    )


def test_volatility_analysis_compares_current_and_historical_performance() -> None:
    result = analyze_volatility_model(
        sample_prices(),
        horizon=5,
        paths=200,
        training_window=160,
        seed=11,
    )

    assert result.current_comparison["model"].tolist() == [
        "Fixed historical bootstrap",
        "EWMA volatility-aware",
    ]
    assert result.adaptive_price_paths.shape == (200, 6)
    assert result.baseline_backtest.observations == result.adaptive_backtest.observations
    assert result.current_annualized_volatility > 0
    assert len(result.volatility_history) == 320


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.5, 1.1])
def test_ewma_filter_rejects_invalid_decay(decay: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        ewma_filter(sample_returns().to_numpy(), decay=decay)
