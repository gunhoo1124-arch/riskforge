import numpy as np
import pandas as pd
import pytest

from riskforge.ml_volatility import (
    DEFAULT_ALPHAS,
    FEATURE_NAMES,
    analyze_ml_volatility,
    build_ml_volatility_dataset,
    walk_forward_ridge_forecast,
)


def sample_returns(size: int = 800, seed: int = 91) -> pd.Series:
    rng = np.random.default_rng(seed)
    volatility = np.empty(size)
    volatility[0] = 0.01
    returns = np.empty(size)
    returns[0] = rng.normal(0, volatility[0])
    for index in range(1, size):
        volatility[index] = 0.002 + 0.82 * volatility[index - 1] + 0.12 * abs(returns[index - 1])
        returns[index] = rng.normal(0.0002, volatility[index])
    return pd.Series(returns, index=pd.bdate_range("2022-01-03", periods=size))


def sample_prices(size: int = 801) -> pd.Series:
    returns = sample_returns(size - 1)
    prices = 100 * np.exp(np.r_[0.0, np.cumsum(returns.to_numpy())])
    return pd.Series(prices, index=pd.bdate_range("2022-01-03", periods=size))


def test_dataset_uses_non_overlapping_future_periods() -> None:
    returns = sample_returns()
    dataset = build_ml_volatility_dataset(returns, horizon=5)

    forecast_positions = returns.index.get_indexer(dataset["forecast_date"])
    outcome_positions = returns.index.get_indexer(dataset["outcome_date"])
    assert (outcome_positions - forecast_positions == 5).all()
    assert (np.diff(outcome_positions) == 5).all()
    assert dataset[list(FEATURE_NAMES)].notna().all().all()


def test_appending_future_data_does_not_change_existing_examples() -> None:
    original = sample_returns(size=500)
    extended = pd.concat(
        [
            original,
            pd.Series(
                np.full(100, 0.20),
                index=pd.bdate_range(original.index[-1] + pd.offsets.BDay(1), periods=100),
            ),
        ]
    )

    original_dataset = build_ml_volatility_dataset(original, horizon=5)
    extended_dataset = build_ml_volatility_dataset(extended, horizon=5)

    pd.testing.assert_frame_equal(
        original_dataset,
        extended_dataset.iloc[: len(original_dataset)].reset_index(drop=True),
    )


def test_walk_forward_forecast_is_strictly_chronological() -> None:
    dataset = build_ml_volatility_dataset(sample_returns(), horizon=5)
    first = walk_forward_ridge_forecast(dataset)
    changed = dataset.copy()
    changed.loc[first.initial_training_samples + 1 :, "actual_volatility"] *= 10
    second = walk_forward_ridge_forecast(changed)

    assert first.selected_alpha in DEFAULT_ALPHAS
    assert first.testing_samples == len(dataset) - first.initial_training_samples
    assert first.predictions.iloc[0]["ml_predicted_volatility"] == pytest.approx(
        second.predictions.iloc[0]["ml_predicted_volatility"]
    )


def test_ml_analysis_is_reproducible_and_builds_paths() -> None:
    first = analyze_ml_volatility(sample_prices(), horizon=5, paths=200, seed=12)
    second = analyze_ml_volatility(sample_prices(), horizon=5, paths=200, seed=12)

    np.testing.assert_array_equal(first.ml_price_paths, second.ml_price_paths)
    assert first.ml_price_paths.shape == (200, 6)
    assert first.current_risk_comparison["model"].tolist() == [
        "EWMA-only volatility",
        "Ridge ML + EWMA paths",
    ]
    assert first.evaluation.testing_samples > 0
    assert np.isfinite(first.evaluation.ml_mae)
    assert first.current_forecast_annualized_volatility > 0


def test_ml_dataset_rejects_too_few_non_overlapping_examples() -> None:
    with pytest.raises(ValueError, match="non-overlapping examples"):
        build_ml_volatility_dataset(sample_returns(size=200), horizon=20)
