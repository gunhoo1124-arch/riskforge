import numpy as np
import pytest

from riskforge.simulation import (
    returns_to_price_paths,
    simulate_bootstrap,
    simulate_iid_bootstrap,
    simulate_moving_block_bootstrap,
)


def test_bootstrap_shape_and_reproducibility() -> None:
    returns = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

    first = simulate_iid_bootstrap(returns, horizon=7, paths=25, seed=123)
    second = simulate_iid_bootstrap(returns, horizon=7, paths=25, seed=123)

    assert first.shape == (25, 7)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_change_bootstrap_sample() -> None:
    returns = np.array([-0.02, 0.0, 0.02])

    first = simulate_iid_bootstrap(returns, horizon=5, paths=10, seed=1)
    second = simulate_iid_bootstrap(returns, horizon=5, paths=10, seed=2)

    assert not np.array_equal(first, second)


def test_returns_to_price_paths_includes_initial_price() -> None:
    simulated = np.array([[np.log(1.1), np.log(0.9)]])

    paths = returns_to_price_paths(simulated, initial_price=100.0)

    assert paths.shape == (1, 3)
    np.testing.assert_allclose(paths[0], [100.0, 110.0, 99.0])


def test_bootstrap_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        simulate_iid_bootstrap(np.ones((2, 2)))


def test_moving_block_bootstrap_is_reproducible_and_keeps_blocks_consecutive() -> None:
    returns = np.arange(20, dtype=float)

    first = simulate_moving_block_bootstrap(
        returns,
        horizon=8,
        paths=25,
        block_size=3,
        seed=19,
    )
    second = simulate_moving_block_bootstrap(
        returns,
        horizon=8,
        paths=25,
        block_size=3,
        seed=19,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (25, 8)
    within_block_positions = [1, 2, 4, 5, 7]
    np.testing.assert_array_equal(
        first[:, within_block_positions] - first[:, np.array(within_block_positions) - 1],
        np.ones((25, len(within_block_positions))),
    )


def test_one_day_blocks_match_iid_bootstrap() -> None:
    returns = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

    iid = simulate_iid_bootstrap(returns, horizon=7, paths=20, seed=8)
    blocked = simulate_moving_block_bootstrap(
        returns,
        horizon=7,
        paths=20,
        block_size=1,
        seed=8,
    )

    np.testing.assert_array_equal(iid, blocked)


def test_block_size_is_capped_at_short_forecast_horizon() -> None:
    returns = np.arange(10, dtype=float)

    simulated = simulate_moving_block_bootstrap(
        returns,
        horizon=1,
        paths=1_000,
        block_size=5,
        seed=5,
    )

    assert simulated.min() == 0
    assert simulated.max() == 9


def test_bootstrap_dispatcher_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown bootstrap method"):
        simulate_bootstrap(np.arange(10), method="made-up")


def test_moving_block_rejects_block_larger_than_history() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        simulate_moving_block_bootstrap(np.arange(5), block_size=6)
