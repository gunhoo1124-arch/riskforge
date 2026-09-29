import numpy as np
import pandas as pd
import pytest

from riskforge.robustness import analyze_model_robustness, model_configurations


def sample_prices(size: int = 360) -> pd.Series:
    rng = np.random.default_rng(71)
    log_returns = rng.normal(0.0002, 0.012, size=size - 1)
    values = 100 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=size))


def test_model_configurations_remove_equivalent_block_sizes() -> None:
    configurations = model_configurations(horizon=5, block_sizes=(3, 5, 10, 20))

    assert [configuration.label for configuration in configurations] == [
        "Independent days (IID)",
        "Moving blocks: 3 days",
        "Moving blocks: 5 days",
    ]


def test_robustness_compares_models_backtests_and_lookbacks() -> None:
    result = analyze_model_robustness(
        sample_prices(),
        horizon=5,
        paths=200,
        block_sizes=(3, 5),
        selected_block_size=5,
        seed=9,
    )

    assert len(result.model_comparison) == 3
    assert len(result.backtest_comparison) == 3
    assert result.backtest_comparison["observations"].nunique() == 1
    assert result.lookback_comparison["lookback"].tolist() == [252, 359]
    assert result.model_comparison["var_95"].notna().all()
    assert result.evaluation_returns == 180


def test_robustness_is_reproducible() -> None:
    prices = sample_prices(280)

    first = analyze_model_robustness(prices, horizon=5, paths=150, seed=12)
    second = analyze_model_robustness(prices, horizon=5, paths=150, seed=12)

    pd.testing.assert_frame_equal(first.model_comparison, second.model_comparison)
    pd.testing.assert_frame_equal(first.backtest_comparison, second.backtest_comparison)
    pd.testing.assert_frame_equal(first.lookback_comparison, second.lookback_comparison)


def test_robustness_rejects_horizon_too_long_for_history() -> None:
    with pytest.raises(ValueError, match="Not enough history"):
        analyze_model_robustness(sample_prices(120), horizon=60, paths=100)
