"""Historical-bootstrap simulation."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class BootstrapMethod(StrEnum):
    """Supported historical resampling methods."""

    IID = "iid"
    MOVING_BLOCK = "moving-block"


def _validated_returns(historical_returns: ArrayLike) -> FloatArray:
    values = np.asarray(historical_returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("Historical returns must be a one-dimensional array.")
    if values.size < 2:
        raise ValueError("At least two historical returns are required.")
    if not np.isfinite(values).all():
        raise ValueError("Historical returns must contain only finite values.")
    return values


def _validate_simulation_inputs(
    historical_returns: ArrayLike,
    horizon: int,
    paths: int,
    seed: int,
) -> FloatArray:
    """Validate inputs shared by the bootstrap simulators."""
    values = _validated_returns(historical_returns)
    if horizon < 1:
        raise ValueError("Horizon must be at least 1 trading day.")
    if paths < 1:
        raise ValueError("Paths must be at least 1.")
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    return values


def normalize_bootstrap_method(method: BootstrapMethod | str) -> BootstrapMethod:
    """Return a validated bootstrap method from an enum or command-line value."""
    try:
        return BootstrapMethod(method)
    except ValueError as exc:
        choices = ", ".join(item.value for item in BootstrapMethod)
        raise ValueError(f"Unknown bootstrap method '{method}'. Choose one of: {choices}.") from exc


def simulate_iid_bootstrap(
    historical_returns: ArrayLike,
    horizon: int = 20,
    paths: int = 10_000,
    seed: int = 42,
) -> FloatArray:
    """Sample historical log returns independently with replacement.

    Returns an array shaped ``(paths, horizon)``. Supplying the same inputs and
    seed always produces identical output.
    """
    values = _validate_simulation_inputs(historical_returns, horizon, paths, seed)

    rng = np.random.default_rng(seed)
    return rng.choice(values, size=(paths, horizon), replace=True)


def simulate_moving_block_bootstrap(
    historical_returns: ArrayLike,
    horizon: int = 20,
    paths: int = 10_000,
    block_size: int = 5,
    seed: int = 42,
) -> FloatArray:
    """Sample overlapping blocks of consecutive historical log returns.

    Blocks are independently drawn with replacement from every valid historical
    starting position, concatenated, and truncated to ``horizon``. The effective
    block length is capped at the forecast horizon, so a one-day forecast samples
    every historical return uniformly. A block size of one is exactly the IID model.
    """
    values = _validate_simulation_inputs(historical_returns, horizon, paths, seed)
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("Block size must be a positive integer.")
    if block_size > len(values):
        raise ValueError("Block size cannot exceed the number of historical returns.")
    if block_size == 1:
        return simulate_iid_bootstrap(values, horizon=horizon, paths=paths, seed=seed)

    effective_block_size = min(block_size, horizon)
    rng = np.random.default_rng(seed)
    result = np.empty((paths, horizon), dtype=float)
    position = 0
    while position < horizon:
        remaining = horizon - position
        current_size = min(effective_block_size, remaining)
        starts = rng.integers(
            0,
            len(values) - effective_block_size + 1,
            size=paths,
        )
        indices = starts[:, np.newaxis] + np.arange(current_size)
        result[:, position : position + current_size] = values[indices]
        position += current_size
    return result


def simulate_bootstrap(
    historical_returns: ArrayLike,
    horizon: int = 20,
    paths: int = 10_000,
    method: BootstrapMethod | str = BootstrapMethod.MOVING_BLOCK,
    block_size: int = 5,
    seed: int = 42,
) -> FloatArray:
    """Dispatch to a validated historical-bootstrap simulation method."""
    selected_method = normalize_bootstrap_method(method)
    if selected_method is BootstrapMethod.IID:
        return simulate_iid_bootstrap(
            historical_returns,
            horizon=horizon,
            paths=paths,
            seed=seed,
        )
    return simulate_moving_block_bootstrap(
        historical_returns,
        horizon=horizon,
        paths=paths,
        block_size=block_size,
        seed=seed,
    )


def returns_to_price_paths(simulated_returns: ArrayLike, initial_price: float) -> FloatArray:
    """Convert simulated log returns to price paths, including the day-zero price.

    If returns have shape ``(n, h)``, the returned paths have shape ``(n, h + 1)``.
    """
    returns = np.asarray(simulated_returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("Simulated returns must be a two-dimensional array.")
    if returns.shape[0] < 1 or returns.shape[1] < 1:
        raise ValueError("Simulated returns cannot be empty.")
    if not np.isfinite(returns).all():
        raise ValueError("Simulated returns must contain only finite values.")
    if not np.isfinite(initial_price) or initial_price <= 0:
        raise ValueError("Initial price must be a positive finite number.")

    cumulative_log_returns = np.cumsum(returns, axis=1)
    future_prices = initial_price * np.exp(cumulative_log_returns)
    initial_column = np.full((returns.shape[0], 1), initial_price, dtype=float)
    return np.concatenate((initial_column, future_prices), axis=1)
