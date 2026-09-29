"""Market-risk statistics for simulated price paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RiskSummary:
    """Summary statistics expressed as decimal fractions."""

    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    probability_of_loss: float
    probability_exceeding_threshold: float
    mean_maximum_drawdown: float
    maximum_drawdown_95: float


def _validated_paths(price_paths: ArrayLike) -> FloatArray:
    paths = np.asarray(price_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 2:
        raise ValueError("Price paths must have shape (paths, at least 2 time points).")
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError("Price paths must contain only positive finite values.")
    return paths


def terminal_returns(price_paths: ArrayLike) -> FloatArray:
    """Calculate each path's terminal simple return relative to day zero."""
    paths = _validated_paths(price_paths)
    return paths[:, -1] / paths[:, 0] - 1.0


def value_at_risk(losses: ArrayLike, confidence: float) -> float:
    """Return the empirical loss quantile at ``confidence``.

    Losses use the convention ``loss = -return``. A positive result denotes a
    loss, while a negative result denotes a gain at the selected quantile.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Losses must be a non-empty, finite one-dimensional array.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between 0 and 1.")
    return float(np.quantile(values, confidence))


def expected_shortfall(losses: ArrayLike, confidence: float) -> float:
    """Average empirical losses greater than or equal to VaR."""
    values = np.asarray(losses, dtype=float)
    var = value_at_risk(values, confidence)
    tail = values[values >= var]
    return float(tail.mean())


def maximum_drawdowns(price_paths: ArrayLike) -> FloatArray:
    """Calculate each path's maximum peak-to-trough drawdown as a positive value."""
    paths = _validated_paths(price_paths)
    running_peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = 1.0 - paths / running_peaks
    return np.max(drawdowns, axis=1)


def summarize_risk(price_paths: ArrayLike, loss_threshold: float = 0.10) -> RiskSummary:
    """Calculate terminal-loss and drawdown statistics for simulated paths."""
    if not 0 <= loss_threshold <= 1:
        raise ValueError("Loss threshold must be between 0 and 1.")

    returns = terminal_returns(price_paths)
    losses = -returns
    drawdowns = maximum_drawdowns(price_paths)

    return RiskSummary(
        var_95=value_at_risk(losses, 0.95),
        var_99=value_at_risk(losses, 0.99),
        expected_shortfall_95=expected_shortfall(losses, 0.95),
        expected_shortfall_99=expected_shortfall(losses, 0.99),
        probability_of_loss=float(np.mean(returns < 0)),
        probability_exceeding_threshold=float(np.mean(losses >= loss_threshold)),
        mean_maximum_drawdown=float(drawdowns.mean()),
        maximum_drawdown_95=float(np.quantile(drawdowns, 0.95)),
    )
