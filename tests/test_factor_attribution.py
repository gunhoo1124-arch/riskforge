import numpy as np
import pandas as pd
import pytest

from riskforge.factor_attribution import analyze_factor_exposures, apply_factor_stress


def _synthetic_factor_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(15)
    factor_returns = rng.normal(0.0, [0.01, 0.006], size=(300, 2))
    asset_returns = np.column_stack(
        (
            1.2 * factor_returns[:, 0] + 0.2 * factor_returns[:, 1],
            -0.3 * factor_returns[:, 0] + 0.8 * factor_returns[:, 1],
        )
    )
    dates = pd.bdate_range("2024-01-02", periods=301)
    factor_prices = pd.DataFrame(
        100 * np.exp(np.vstack((np.zeros(2), np.cumsum(factor_returns, axis=0)))),
        index=dates,
        columns=["MKT", "BOND"],
    )
    asset_prices = pd.DataFrame(
        100 * np.exp(np.vstack((np.zeros(2), np.cumsum(asset_returns, axis=0)))),
        index=dates,
        columns=["AAA", "BBB"],
    )
    return asset_prices, factor_prices


def test_factor_analysis_recovers_known_betas_and_portfolio_exposure() -> None:
    assets, factors = _synthetic_factor_prices()

    result = analyze_factor_exposures(assets, factors, [0.6, 0.4])

    np.testing.assert_allclose(result.asset_betas.loc["AAA"], [1.2, 0.2], atol=1e-10)
    np.testing.assert_allclose(result.asset_betas.loc["BBB"], [-0.3, 0.8], atol=1e-10)
    np.testing.assert_allclose(result.portfolio_betas, [0.6, 0.44], atol=1e-10)
    assert result.portfolio_r_squared == pytest.approx(1.0)
    assert result.observations == 300


def test_variance_attribution_reconciles_and_reports_residual() -> None:
    assets, factors = _synthetic_factor_prices()

    result = analyze_factor_exposures(assets, factors, [0.5, 0.5])

    assert result.variance_attribution["share_of_total_variance"].sum() == pytest.approx(1.0)
    residual = result.variance_attribution.iloc[-1]
    assert residual["driver"] == "Unexplained / holding-specific"
    assert residual["annualized_variance_contribution"] == pytest.approx(0.0, abs=1e-20)


def test_factor_stress_contributions_reconcile_across_views() -> None:
    assets, factors = _synthetic_factor_prices()
    attribution = analyze_factor_exposures(assets, factors, [0.6, 0.4])

    stress = apply_factor_stress(
        attribution,
        {"MKT": -0.20, "BOND": 0.05},
        initial_value=100_000,
    )

    expected = 0.6 * -0.20 + 0.44 * 0.05
    assert stress.portfolio_implied_return == pytest.approx(expected)
    assert stress.factor_contributions["portfolio_return_contribution"].sum() == pytest.approx(
        expected
    )
    assert stress.holding_impacts["portfolio_return_contribution"].sum() == pytest.approx(expected)
    assert stress.portfolio_implied_change == pytest.approx(expected * 100_000)


def test_factor_analysis_rejects_perfectly_redundant_proxies() -> None:
    assets, factors = _synthetic_factor_prices()
    factors["COPY"] = factors["MKT"]

    with pytest.raises(ValueError, match="perfectly redundant"):
        analyze_factor_exposures(assets, factors, [0.5, 0.5])


def test_factor_stress_requires_exact_fitted_factor_set() -> None:
    assets, factors = _synthetic_factor_prices()
    attribution = analyze_factor_exposures(assets, factors, [0.5, 0.5])

    with pytest.raises(ValueError, match="exactly one shock"):
        apply_factor_stress(attribution, {"MKT": -0.20}, initial_value=100_000)
