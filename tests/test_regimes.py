import numpy as np
import pandas as pd
import pytest

from riskforge.regimes import MarketRegime, analyze_market_regimes


def prices_from_log_returns(log_returns: np.ndarray) -> pd.Series:
    values = 100 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)))


def test_regime_analysis_classifies_turbulent_decline() -> None:
    calm_advance = np.full(100, 0.001)
    moderate_decline = np.tile([-0.02, 0.015], 20)
    turbulent_decline = np.tile([-0.05, 0.03], 20)
    prices = prices_from_log_returns(
        np.concatenate([calm_advance, moderate_decline, turbulent_decline])
    )

    result = analyze_market_regimes(prices, window=20, forward_horizon=5)

    assert result.current_regime is MarketRegime.TURBULENT_DECLINE
    assert result.current_annualized_volatility > result.volatility_threshold
    assert result.current_rolling_return < 0
    assert result.outcome_observations == len(prices) - 20 - 5


def test_forward_outcome_uses_only_prices_after_classification_date() -> None:
    prices = pd.Series(
        np.linspace(100.0, 160.0, 80),
        index=pd.bdate_range("2024-01-01", periods=80),
    )

    result = analyze_market_regimes(prices, window=20, forward_horizon=5)
    first = result.history.iloc[0]
    start_position = prices.index.get_loc(first["date"])
    expected = prices.iloc[start_position + 5] / prices.iloc[start_position] - 1

    assert first["forward_return"] == pytest.approx(expected)


def test_regime_summary_probabilities_and_tail_metrics_are_valid() -> None:
    rng = np.random.default_rng(7)
    prices = prices_from_log_returns(rng.normal(0.0002, 0.015, size=180))

    result = analyze_market_regimes(prices, window=30, forward_horizon=10)

    assert result.summary["observations"].sum() == result.outcome_observations
    assert result.summary["sample_share"].sum() == pytest.approx(1.0)
    assert result.summary["probability_of_loss"].between(0, 1).all()
    assert (result.summary["expected_shortfall_95"] >= result.summary["var_95"]).all()


def test_regime_analysis_is_deterministic() -> None:
    rng = np.random.default_rng(13)
    prices = prices_from_log_returns(rng.normal(0.0, 0.01, size=120))

    first = analyze_market_regimes(prices, window=20, forward_horizon=5)
    second = analyze_market_regimes(prices, window=20, forward_horizon=5)

    pd.testing.assert_frame_equal(first.history, second.history)
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_historical_label_does_not_use_later_prices() -> None:
    rng = np.random.default_rng(21)
    original = prices_from_log_returns(rng.normal(0.0, 0.01, size=120))
    changed_future = original.copy()
    changed_future.iloc[71:] *= np.linspace(1.2, 0.7, len(changed_future) - 71)

    original_result = analyze_market_regimes(original, window=20, forward_horizon=5)
    changed_result = analyze_market_regimes(changed_future, window=20, forward_horizon=5)
    comparison_date = original.index[70]
    original_row = original_result.history.set_index("date").loc[comparison_date]
    changed_row = changed_result.history.set_index("date").loc[comparison_date]

    assert original_row["regime"] == changed_row["regime"]
    assert original_row["volatility_threshold"] == pytest.approx(
        changed_row["volatility_threshold"]
    )


def test_regime_analysis_rejects_insufficient_history() -> None:
    prices = prices_from_log_returns(np.zeros(24))

    with pytest.raises(ValueError, match="Not enough prices"):
        analyze_market_regimes(prices, window=20, forward_horizon=5)
