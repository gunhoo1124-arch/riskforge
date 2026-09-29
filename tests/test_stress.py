import numpy as np
import pandas as pd
import pytest

from riskforge.stress import analyze_historical_stress, historical_exceedance_rate


def price_series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)))


def test_historical_stress_preserves_realized_horizon_returns() -> None:
    prices = price_series([100.0, 110.0, 99.0, 79.2, 87.12])

    result = analyze_historical_stress(prices, horizon=2, loss_threshold=0.15)

    np.testing.assert_allclose(result.periods["return"], [-0.01, -0.28, -0.12])
    assert result.observations == 3
    assert result.worst_loss == pytest.approx(0.28)
    assert result.worst_start_date == prices.index[1]
    assert result.worst_end_date == prices.index[3]
    assert result.probability_exceeding_threshold == pytest.approx(1 / 3)


def test_worst_episode_table_does_not_repeat_overlapping_windows() -> None:
    prices = price_series([100, 98, 96, 70, 60, 58, 62, 61, 40, 39, 42])

    result = analyze_historical_stress(prices, horizon=2, top_n=3)

    positions = {date: index for index, date in enumerate(prices.index)}
    windows = [
        (positions[row.start_date], positions[row.end_date])
        for row in result.worst_episodes.itertuples()
    ]
    for index, (start, end) in enumerate(windows):
        for other_start, other_end in windows[index + 1 :]:
            assert end <= other_start or start >= other_end


def test_historical_exceedance_rate_uses_strictly_worse_losses() -> None:
    result = analyze_historical_stress(
        price_series([100.0, 80.0, 72.0, 86.4]),
        horizon=1,
    )

    assert historical_exceedance_rate(result, 0.10) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("prices", "horizon", "message"),
    [
        (price_series([100.0, 101.0]), 2, "Not enough prices"),
        (price_series([100.0, 101.0]), 0, "positive integer"),
        (price_series([100.0, -1.0, 101.0]), 1, "positive finite"),
    ],
)
def test_historical_stress_validates_inputs(
    prices: pd.Series,
    horizon: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_historical_stress(prices, horizon)
