"""Leakage-aware ridge regression for horizon volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from riskforge.metrics import RiskSummary, summarize_risk
from riskforge.returns import calculate_log_returns
from riskforge.simulation import BootstrapMethod, returns_to_price_paths
from riskforge.volatility import (
    TRADING_DAYS_PER_YEAR,
    ewma_filter,
    simulate_ewma_filtered_bootstrap,
)

FEATURE_NAMES = (
    "log_volatility_5",
    "log_volatility_20",
    "log_volatility_63",
    "log_ewma_volatility",
    "absolute_return_1",
    "downside_volatility_5",
    "trend_20",
    "log_volatility_ratio_5_to_20",
)
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
MINIMUM_DATASET_ROWS = 30
VOLATILITY_FLOOR = 1e-6
VOLATILITY_CEILING = 5.0


@dataclass(frozen=True, slots=True)
class RidgeEvaluation:
    """Nested time-series evaluation of ridge forecasts against EWMA."""

    predictions: pd.DataFrame
    selected_alpha: float
    initial_training_samples: int
    testing_samples: int
    ml_mae: float
    ewma_mae: float
    ml_rmse: float
    ewma_rmse: float


@dataclass(frozen=True, slots=True)
class MlVolatilityResult:
    """Out-of-sample ML evaluation and ML-scaled Monte Carlo scenarios."""

    evaluation: RidgeEvaluation
    current_forecast_annualized_volatility: float
    current_ewma_annualized_volatility: float
    current_risk_comparison: pd.DataFrame
    ml_price_paths: np.ndarray
    coefficients: pd.DataFrame
    current_features: pd.DataFrame
    dataset_rows: int
    horizon: int
    paths: int
    prediction_was_clipped: bool


def _realized_volatility(returns: np.ndarray) -> float:
    """Return zero-mean RMS volatility annualized with 252 trading days."""
    return float(np.sqrt(np.mean(np.square(returns))) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _log_volatility(value: float) -> float:
    """Return a numerically safe log volatility."""
    return float(np.log(max(value, VOLATILITY_FLOOR)))


def _feature_values(values: np.ndarray, origin: int) -> dict[str, float]:
    """Build forecast-origin features using returns no later than ``origin``."""
    history = values[: origin + 1]
    if len(history) < 63:
        raise ValueError("At least 63 returns are required to build ML volatility features.")

    vol_5 = _realized_volatility(history[-5:])
    vol_20 = _realized_volatility(history[-20:])
    vol_63 = _realized_volatility(history[-63:])
    ewma_volatility = ewma_filter(history).next_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    downside = np.minimum(history[-5:], 0.0)
    downside_volatility = _realized_volatility(downside)
    return {
        "log_volatility_5": _log_volatility(vol_5),
        "log_volatility_20": _log_volatility(vol_20),
        "log_volatility_63": _log_volatility(vol_63),
        "log_ewma_volatility": _log_volatility(ewma_volatility),
        "absolute_return_1": float(abs(history[-1]) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "downside_volatility_5": downside_volatility,
        "trend_20": float(history[-20:].sum()),
        "log_volatility_ratio_5_to_20": float(
            np.log(max(vol_5, VOLATILITY_FLOOR) / max(vol_20, VOLATILITY_FLOOR))
        ),
    }


def build_ml_volatility_dataset(
    log_returns: pd.Series,
    horizon: int = 5,
) -> pd.DataFrame:
    """Build non-overlapping, chronologically ordered volatility examples.

    Each feature row uses returns through its forecast date. Its target uses only
    the following ``horizon`` returns. Consecutive targets do not overlap.
    """
    if not isinstance(log_returns, pd.Series):
        raise TypeError("Log returns must be provided as a pandas Series.")
    values = log_returns.to_numpy(dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Log returns must be a finite one-dimensional series.")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")
    if len(values) < 63 + horizon:
        raise ValueError("At least 63 feature returns plus one forecast horizon are required.")

    rows: list[dict[str, object]] = []
    for origin in range(62, len(values) - horizon, horizon):
        future = values[origin + 1 : origin + horizon + 1]
        target_volatility = _realized_volatility(future)
        row: dict[str, object] = {
            "forecast_date": log_returns.index[origin],
            "outcome_date": log_returns.index[origin + horizon],
            **_feature_values(values, origin),
            "actual_volatility": target_volatility,
            "log_actual_volatility": _log_volatility(target_volatility),
        }
        rows.append(row)

    dataset = pd.DataFrame(rows)
    if len(dataset) < MINIMUM_DATASET_ROWS:
        raise ValueError(
            f"This ML horizon produced only {len(dataset)} non-overlapping examples; at least "
            f"{MINIMUM_DATASET_ROWS} are required. Choose a shorter horizon or longer history."
        )
    return dataset


def _ridge_pipeline(alpha: float) -> Pipeline:
    """Build a leakage-safe scaling and ridge-regression pipeline."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _select_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    alphas: tuple[float, ...],
) -> float:
    """Choose ridge strength using only expanding splits inside initial training."""
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge alpha candidates must be positive finite numbers.")
    split_count = min(4, max(2, len(features) // 6))
    splitter = TimeSeriesSplit(n_splits=split_count)
    scores: dict[float, float] = {}
    for alpha in alphas:
        fold_errors: list[float] = []
        for training_indices, validation_indices in splitter.split(features):
            model = _ridge_pipeline(alpha)
            model.fit(features[training_indices], targets[training_indices])
            predictions = model.predict(features[validation_indices])
            fold_errors.append(float(np.mean(np.abs(targets[validation_indices] - predictions))))
        scores[float(alpha)] = float(np.mean(fold_errors))
    return min(scores, key=lambda alpha: (scores[alpha], alpha))


def walk_forward_ridge_forecast(
    dataset: pd.DataFrame,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> RidgeEvaluation:
    """Tune on initial history, then refit and predict one future row at a time."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Dataset must be provided as a pandas DataFrame.")
    required = {*FEATURE_NAMES, "forecast_date", "outcome_date", "actual_volatility"}
    missing = required.difference(dataset.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"ML volatility dataset is missing columns: {names}.")
    if len(dataset) < MINIMUM_DATASET_ROWS:
        raise ValueError(f"At least {MINIMUM_DATASET_ROWS} examples are required.")

    features = dataset.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    actual_volatility = dataset["actual_volatility"].to_numpy(dtype=float)
    targets = np.log(np.maximum(actual_volatility, VOLATILITY_FLOOR))
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("ML volatility features and targets must be finite.")

    initial_training_samples = max(20, int(len(dataset) * 0.60))
    if len(dataset) - initial_training_samples < 8:
        raise ValueError("At least eight untouched testing examples are required.")
    selected_alpha = _select_alpha(
        features[:initial_training_samples],
        targets[:initial_training_samples],
        alphas,
    )

    rows: list[dict[str, object]] = []
    for index in range(initial_training_samples, len(dataset)):
        model = _ridge_pipeline(selected_alpha)
        model.fit(features[:index], targets[:index])
        raw_log_prediction = float(model.predict(features[index : index + 1])[0])
        ml_prediction = float(
            np.exp(
                np.clip(
                    raw_log_prediction,
                    np.log(VOLATILITY_FLOOR),
                    np.log(VOLATILITY_CEILING),
                )
            )
        )
        ewma_prediction = float(np.exp(dataset.iloc[index]["log_ewma_volatility"]))
        actual = float(actual_volatility[index])
        rows.append(
            {
                "forecast_date": dataset.iloc[index]["forecast_date"],
                "outcome_date": dataset.iloc[index]["outcome_date"],
                "actual_volatility": actual,
                "ml_predicted_volatility": ml_prediction,
                "ewma_predicted_volatility": ewma_prediction,
                "ml_absolute_error": abs(actual - ml_prediction),
                "ewma_absolute_error": abs(actual - ewma_prediction),
            }
        )

    predictions = pd.DataFrame(rows)
    ml_errors = predictions["actual_volatility"] - predictions["ml_predicted_volatility"]
    ewma_errors = predictions["actual_volatility"] - predictions["ewma_predicted_volatility"]
    return RidgeEvaluation(
        predictions=predictions,
        selected_alpha=selected_alpha,
        initial_training_samples=initial_training_samples,
        testing_samples=len(predictions),
        ml_mae=float(np.mean(np.abs(ml_errors))),
        ewma_mae=float(np.mean(np.abs(ewma_errors))),
        ml_rmse=float(np.sqrt(np.mean(np.square(ml_errors)))),
        ewma_rmse=float(np.sqrt(np.mean(np.square(ewma_errors)))),
    )


def _risk_row(label: str, summary: RiskSummary) -> dict[str, object]:
    """Convert current scenario risk into a comparison row."""
    return {
        "model": label,
        "var_95": summary.var_95,
        "var_99": summary.var_99,
        "expected_shortfall_95": summary.expected_shortfall_95,
        "expected_shortfall_99": summary.expected_shortfall_99,
        "probability_of_loss": summary.probability_of_loss,
        "maximum_drawdown_95": summary.maximum_drawdown_95,
    }


def analyze_ml_volatility(
    prices: pd.Series,
    *,
    horizon: int = 5,
    paths: int = 2_000,
    seed: int = 42,
    decay: float = 0.94,
    loss_threshold: float = 0.10,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
) -> MlVolatilityResult:
    """Evaluate ridge forecasts and use the current forecast in Monte Carlo paths."""
    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be provided as a pandas Series.")
    if not isinstance(paths, int) or isinstance(paths, bool) or paths < 100:
        raise ValueError("At least 100 paths are required for ML volatility analysis.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    if not 0 <= loss_threshold <= 1:
        raise ValueError("Loss threshold must be between 0 and 1.")

    log_returns = calculate_log_returns(prices)
    dataset = build_ml_volatility_dataset(log_returns, horizon=horizon)
    evaluation = walk_forward_ridge_forecast(dataset)
    values = log_returns.to_numpy(dtype=float)
    current_feature_values = _feature_values(values, len(values) - 1)
    current_features = np.array(
        [[current_feature_values[name] for name in FEATURE_NAMES]],
        dtype=float,
    )
    full_features = dataset.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    full_targets = dataset["log_actual_volatility"].to_numpy(dtype=float)
    final_model = _ridge_pipeline(evaluation.selected_alpha)
    final_model.fit(full_features, full_targets)
    raw_current_log_forecast = float(final_model.predict(current_features)[0])
    clipped_current_log_forecast = float(
        np.clip(
            raw_current_log_forecast,
            np.log(VOLATILITY_FLOOR),
            np.log(VOLATILITY_CEILING),
        )
    )
    current_forecast = float(np.exp(clipped_current_log_forecast))
    current_ewma = ewma_filter(values, decay=decay).next_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    ewma_simulation = simulate_ewma_filtered_bootstrap(
        values,
        horizon=horizon,
        paths=paths,
        decay=decay,
        method=method,
        block_size=block_size,
        seed=seed,
    )
    ml_simulation = simulate_ewma_filtered_bootstrap(
        values,
        horizon=horizon,
        paths=paths,
        decay=decay,
        method=method,
        block_size=block_size,
        seed=seed,
        starting_volatility=current_forecast / np.sqrt(TRADING_DAYS_PER_YEAR),
    )
    initial_price = float(prices.iloc[-1])
    ewma_paths = returns_to_price_paths(ewma_simulation.simulated_returns, initial_price)
    ml_paths = returns_to_price_paths(ml_simulation.simulated_returns, initial_price)
    ewma_summary = summarize_risk(ewma_paths, loss_threshold)
    ml_summary = summarize_risk(ml_paths, loss_threshold)

    scaler = final_model.named_steps["scale"]
    ridge = final_model.named_steps["ridge"]
    standardized_coefficients = np.asarray(ridge.coef_, dtype=float)
    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "standardized_coefficient": standardized_coefficients,
        }
    ).sort_values("standardized_coefficient", key=np.abs, ascending=False)
    current_feature_frame = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "value": current_features[0],
            "training_mean": np.asarray(scaler.mean_, dtype=float),
        }
    )

    return MlVolatilityResult(
        evaluation=evaluation,
        current_forecast_annualized_volatility=current_forecast,
        current_ewma_annualized_volatility=float(current_ewma),
        current_risk_comparison=pd.DataFrame(
            [
                _risk_row("EWMA-only volatility", ewma_summary),
                _risk_row("Ridge ML + EWMA paths", ml_summary),
            ]
        ),
        ml_price_paths=ml_paths,
        coefficients=coefficients.reset_index(drop=True),
        current_features=current_feature_frame,
        dataset_rows=len(dataset),
        horizon=horizon,
        paths=paths,
        prediction_was_clipped=not np.isclose(
            raw_current_log_forecast,
            clipped_current_log_forecast,
        ),
    )
