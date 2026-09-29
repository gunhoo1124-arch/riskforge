"""Transparent proxy-factor exposure and stress attribution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from riskforge.portfolio import validate_weights

FloatArray = NDArray[np.float64]
TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class FactorAttributionResult:
    """Historical linear proxy-factor exposure estimates."""

    holdings: tuple[str, ...]
    factors: tuple[str, ...]
    weights: FloatArray
    asset_betas: pd.DataFrame
    portfolio_betas: pd.Series
    regression_diagnostics: pd.DataFrame
    factor_diagnostics: pd.DataFrame
    factor_correlation: pd.DataFrame
    variance_attribution: pd.DataFrame
    observations: int
    condition_number: float
    portfolio_r_squared: float
    portfolio_residual_annualized_volatility: float


@dataclass(frozen=True, slots=True)
class FactorStressResult:
    """Linear portfolio and holding impacts from user-defined proxy shocks."""

    factor_shocks: pd.Series
    factor_contributions: pd.DataFrame
    holding_impacts: pd.DataFrame
    portfolio_implied_return: float
    portfolio_implied_change: float
    initial_value: float


def _validated_price_frame(prices: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 1 or len(prices) < 3:
        raise ValueError(f"{name} prices need at least three rows and one column.")
    if prices.columns.duplicated().any():
        raise ValueError(f"{name} ticker columns must be unique.")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} prices must contain only finite values.")
    if (numeric <= 0).any().any():
        raise ValueError(f"{name} prices must be strictly positive.")
    return numeric.astype(float)


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Calculated factor-analysis returns contain invalid values.")
    return returns


def _fit_ols(
    factor_values: FloatArray,
    target: FloatArray,
) -> tuple[FloatArray, float, float, FloatArray]:
    design = np.column_stack((np.ones(factor_values.shape[0]), factor_values))
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residuals = target - fitted
    total_sum_squares = float(np.sum((target - target.mean()) ** 2))
    if total_sum_squares <= np.finfo(float).eps:
        raise ValueError("A return series has no variation, so its factor exposure is undefined.")
    residual_sum_squares = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    observations = len(target)
    factor_count = factor_values.shape[1]
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (observations - 1) / (
        observations - factor_count - 1
    )
    return coefficients[1:], float(r_squared), float(adjusted_r_squared), residuals


def _variance_inflation_factors(factors: pd.DataFrame) -> FloatArray:
    values = factors.to_numpy(dtype=float)
    if values.shape[1] == 1:
        return np.ones(1, dtype=float)

    vifs = np.empty(values.shape[1], dtype=float)
    for index in range(values.shape[1]):
        target = values[:, index]
        other_values = np.delete(values, index, axis=1)
        _, r_squared, _, _ = _fit_ols(other_values, target)
        denominator = max(1.0 - r_squared, np.finfo(float).eps)
        vifs[index] = 1.0 / denominator
    return vifs


def analyze_factor_exposures(
    asset_prices: pd.DataFrame,
    factor_prices: pd.DataFrame,
    weights: ArrayLike,
    *,
    minimum_observations: int = 60,
) -> FactorAttributionResult:
    """Estimate historical asset and starting-weight portfolio proxy betas.

    The regression is descriptive and uses daily log returns with an intercept.
    Factor tickers are user-selected traded proxies, not structural economic factors.
    """
    assets = _validated_price_frame(asset_prices, "Holding")
    factors = _validated_price_frame(factor_prices, "Factor")
    if len(factors.columns) > 6:
        raise ValueError("Use no more than six proxy factors in one analysis.")
    if len(set(map(str, factors.columns))) != len(factors.columns):
        raise ValueError("Each factor ticker must be unique.")
    validated_weights = validate_weights(weights, len(assets.columns))
    if minimum_observations < 20:
        raise ValueError("Minimum observations must be at least 20.")

    asset_returns = _log_returns(assets).add_prefix("asset::")
    factor_returns = _log_returns(factors).add_prefix("factor::")
    aligned = asset_returns.join(factor_returns, how="inner").dropna()
    required = max(minimum_observations, 10 * (len(factors.columns) + 1))
    if len(aligned) < required:
        raise ValueError(
            f"Only {len(aligned)} shared return observations were available; "
            f"at least {required} are required for this factor set."
        )

    asset_names = tuple(str(column).upper() for column in assets.columns)
    factor_names = tuple(str(column).upper() for column in factors.columns)
    asset_values = aligned[[f"asset::{column}" for column in assets.columns]].to_numpy()
    factor_values = aligned[[f"factor::{column}" for column in factors.columns]].to_numpy()

    factor_standard_deviations = factor_values.std(axis=0, ddof=1)
    if (factor_standard_deviations <= np.finfo(float).eps).any():
        raise ValueError("Every factor proxy must have changing historical returns.")
    standardized_factors = (factor_values - factor_values.mean(axis=0)) / factor_standard_deviations
    if np.linalg.matrix_rank(standardized_factors) < standardized_factors.shape[1]:
        raise ValueError(
            "The selected factor proxies are perfectly redundant. Remove a duplicate or "
            "near-identical factor."
        )
    condition_number = float(np.linalg.cond(standardized_factors))

    beta_rows: list[FloatArray] = []
    diagnostic_rows: list[dict[str, float | str]] = []
    for index, ticker in enumerate(asset_names):
        betas, r_squared, adjusted_r_squared, residuals = _fit_ols(
            factor_values,
            asset_values[:, index],
        )
        beta_rows.append(betas)
        diagnostic_rows.append(
            {
                "series": ticker,
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
                "annualized_volatility": float(
                    np.std(asset_values[:, index], ddof=1) * np.sqrt(TRADING_DAYS)
                ),
                "residual_annualized_volatility": float(
                    np.std(residuals, ddof=1) * np.sqrt(TRADING_DAYS)
                ),
            }
        )

    asset_betas = pd.DataFrame(beta_rows, index=asset_names, columns=factor_names)
    portfolio_values = asset_values @ validated_weights
    portfolio_beta_values, portfolio_r_squared, portfolio_adjusted_r_squared, residuals = _fit_ols(
        factor_values, portfolio_values
    )
    portfolio_betas = pd.Series(portfolio_beta_values, index=factor_names, name="Portfolio")
    portfolio_residual_volatility = float(np.std(residuals, ddof=1) * np.sqrt(TRADING_DAYS))
    diagnostic_rows.append(
        {
            "series": "Portfolio",
            "r_squared": portfolio_r_squared,
            "adjusted_r_squared": portfolio_adjusted_r_squared,
            "annualized_volatility": float(
                np.std(portfolio_values, ddof=1) * np.sqrt(TRADING_DAYS)
            ),
            "residual_annualized_volatility": portfolio_residual_volatility,
        }
    )

    factor_frame = pd.DataFrame(factor_values, index=aligned.index, columns=factor_names)
    factor_covariance = factor_frame.cov() * TRADING_DAYS
    factor_variance_vector = factor_covariance.to_numpy() @ portfolio_beta_values
    variance_contributions = portfolio_beta_values * factor_variance_vector
    residual_variance = float(np.var(residuals, ddof=1) * TRADING_DAYS)
    decomposed_variance = float(variance_contributions.sum() + residual_variance)
    variance_rows = [
        {
            "driver": factor,
            "annualized_variance_contribution": float(variance_contributions[index]),
            "share_of_total_variance": (
                float(variance_contributions[index] / decomposed_variance)
                if decomposed_variance > 0
                else np.nan
            ),
            "kind": "Proxy factor",
        }
        for index, factor in enumerate(factor_names)
    ]
    variance_rows.append(
        {
            "driver": "Unexplained / holding-specific",
            "annualized_variance_contribution": residual_variance,
            "share_of_total_variance": (
                residual_variance / decomposed_variance if decomposed_variance > 0 else np.nan
            ),
            "kind": "Regression residual",
        }
    )

    factor_diagnostics = pd.DataFrame(
        {
            "factor": factor_names,
            "annualized_volatility": factor_frame.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS),
            "variance_inflation_factor": _variance_inflation_factors(factor_frame),
        }
    )
    return FactorAttributionResult(
        holdings=asset_names,
        factors=factor_names,
        weights=validated_weights,
        asset_betas=asset_betas,
        portfolio_betas=portfolio_betas,
        regression_diagnostics=pd.DataFrame(diagnostic_rows),
        factor_diagnostics=factor_diagnostics,
        factor_correlation=factor_frame.corr(),
        variance_attribution=pd.DataFrame(variance_rows),
        observations=len(aligned),
        condition_number=condition_number,
        portfolio_r_squared=portfolio_r_squared,
        portfolio_residual_annualized_volatility=portfolio_residual_volatility,
    )


def apply_factor_stress(
    attribution: FactorAttributionResult,
    factor_shocks: Mapping[str, float] | pd.Series,
    *,
    initial_value: float,
) -> FactorStressResult:
    """Apply user-defined proxy shocks through historical linear beta estimates."""
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Initial portfolio value must be a positive finite number.")
    shocks = pd.Series(factor_shocks, dtype=float)
    shocks.index = shocks.index.map(lambda value: str(value).upper())
    missing = [factor for factor in attribution.factors if factor not in shocks.index]
    extras = [factor for factor in shocks.index if factor not in attribution.factors]
    if missing or extras:
        raise ValueError("Provide exactly one shock for each fitted factor proxy.")
    shocks = shocks.loc[list(attribution.factors)]
    if not np.isfinite(shocks.to_numpy()).all():
        raise ValueError("Factor shocks must contain only finite values.")
    if (shocks < -1).any() or (shocks > 2).any():
        raise ValueError("Factor shocks must be between -100% and +200%.")

    factor_contribution_values = attribution.portfolio_betas * shocks
    portfolio_implied_return = float(factor_contribution_values.sum())
    factor_contributions = pd.DataFrame(
        {
            "factor": attribution.factors,
            "shock": shocks.to_numpy(),
            "portfolio_beta": attribution.portfolio_betas.to_numpy(),
            "portfolio_return_contribution": factor_contribution_values.to_numpy(),
            "portfolio_dollar_contribution": (
                factor_contribution_values.to_numpy() * initial_value
            ),
        }
    )

    holding_returns = attribution.asset_betas.to_numpy() @ shocks.to_numpy()
    weighted_contributions = attribution.weights * holding_returns
    holding_impacts = pd.DataFrame(
        {
            "ticker": attribution.holdings,
            "starting_weight": attribution.weights,
            "factor_implied_return": holding_returns,
            "portfolio_return_contribution": weighted_contributions,
            "portfolio_dollar_contribution": weighted_contributions * initial_value,
        }
    )
    return FactorStressResult(
        factor_shocks=shocks,
        factor_contributions=factor_contributions,
        holding_impacts=holding_impacts,
        portfolio_implied_return=portfolio_implied_return,
        portfolio_implied_change=portfolio_implied_return * initial_value,
        initial_value=float(initial_value),
    )
