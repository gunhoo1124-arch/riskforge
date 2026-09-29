import numpy as np
import pandas as pd
import pytest

from riskforge.portfolio import (
    analyze_portfolio_risk,
    buy_and_hold_portfolio_paths,
    calculate_portfolio_log_returns,
    investments_to_weights,
    simulate_joint_bootstrap,
    validate_weights,
)


def test_joint_bootstrap_shape_reproducibility_and_row_dependence() -> None:
    returns = np.column_stack((np.arange(10, dtype=float), np.arange(10, dtype=float) + 100))

    first = simulate_joint_bootstrap(returns, horizon=7, paths=30, method="iid", seed=12)
    second = simulate_joint_bootstrap(returns, horizon=7, paths=30, method="iid", seed=12)

    assert first.shape == (30, 7, 2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, :, 1] - first[:, :, 0], np.full((30, 7), 100.0))


def test_joint_moving_blocks_keep_consecutive_market_rows() -> None:
    returns = np.column_stack((np.arange(20, dtype=float), np.arange(20, dtype=float) * -2))

    simulated = simulate_joint_bootstrap(
        returns,
        horizon=8,
        paths=25,
        method="moving-block",
        block_size=3,
        seed=19,
    )

    within_block_positions = [1, 2, 4, 5, 7]
    differences = (
        simulated[:, within_block_positions, 0]
        - simulated[:, np.array(within_block_positions) - 1, 0]
    )
    np.testing.assert_array_equal(differences, np.ones((25, len(within_block_positions))))
    np.testing.assert_array_equal(simulated[:, :, 1], -2 * simulated[:, :, 0])


def test_buy_and_hold_paths_allow_weights_to_drift() -> None:
    simulated = np.array([[[np.log(2.0), np.log(0.5)], [0.0, 0.0]]])

    paths, relatives = buy_and_hold_portfolio_paths(
        simulated,
        weights=[0.5, 0.5],
        initial_value=1_000.0,
    )

    np.testing.assert_allclose(paths, [[1_000.0, 1_250.0, 1_250.0]])
    np.testing.assert_allclose(relatives[0, -1], [2.0, 0.5])


def test_weights_must_sum_to_one_and_be_long_only() -> None:
    with pytest.raises(ValueError, match="100%"):
        validate_weights([0.4, 0.4], 2)
    with pytest.raises(ValueError, match="negative"):
        validate_weights([1.1, -0.1], 2)


def test_investment_amounts_become_normalized_weights() -> None:
    weights, total = investments_to_weights([50_000, 30_000, 20_000], 3)

    np.testing.assert_allclose(weights, [0.5, 0.3, 0.2])
    assert total == 100_000


def test_investment_amounts_must_be_positive_and_match_holdings() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        investments_to_weights([1_000, 0], 2)
    with pytest.raises(ValueError, match="exactly one"):
        investments_to_weights([1_000], 2)


def test_portfolio_analysis_tail_contributions_reconcile_to_es() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(
        loc=[0.0004, 0.0002, 0.0001],
        scale=[0.012, 0.009, 0.006],
        size=(400, 3),
    )
    prices = pd.DataFrame(
        100 * np.exp(np.vstack((np.zeros(3), np.cumsum(returns, axis=0)))),
        index=pd.bdate_range("2024-01-01", periods=401),
        columns=["SPY", "QQQ", "TLT"],
    )

    result = analyze_portfolio_risk(
        prices,
        [0.5, 0.3, 0.2],
        horizon=20,
        paths=2_000,
        initial_value=100_000,
        method="moving-block",
        block_size=5,
        seed=22,
    )

    assert result.portfolio_paths.shape == (2_000, 21)
    assert result.correlation.shape == (3, 3)
    assert result.effective_number_of_assets == pytest.approx(1 / 0.38)
    assert result.tail_contributions["es_contribution"].sum() == pytest.approx(
        result.summary.expected_shortfall_95
    )
    assert result.tail_contributions["es_contribution_dollars"].sum() == pytest.approx(
        result.summary.expected_shortfall_95 * 100_000
    )


def test_calculate_portfolio_log_returns_matches_price_ratios() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, 49.5]},
        index=pd.bdate_range("2026-01-01", periods=3),
    )

    returns = calculate_portfolio_log_returns(prices)

    np.testing.assert_allclose(returns["A"], [np.log(1.1), np.log(1.1)])
    np.testing.assert_allclose(returns["B"], [np.log(0.9), np.log(1.1)])
