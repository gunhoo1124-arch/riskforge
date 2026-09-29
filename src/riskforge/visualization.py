"""Interactive Monte Carlo visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike


def build_fan_chart(
    price_paths: ArrayLike,
    ticker: str,
    sample_paths: int = 25,
) -> go.Figure:
    """Build an interactive fan chart from simulated price paths."""
    paths = np.asarray(price_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 2:
        raise ValueError("Price paths must have shape (paths, at least 2 time points).")
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError("Price paths must contain only positive finite values.")
    if sample_paths < 0:
        raise ValueError("Sample paths cannot be negative.")

    days = np.arange(paths.shape[1])
    p05, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    figure = go.Figure()

    # Draw a selection spread across the simulation array without introducing
    # another source of randomness.
    count = min(sample_paths, paths.shape[0])
    if count:
        indexes = np.linspace(0, paths.shape[0] - 1, count, dtype=int)
        for index in indexes:
            figure.add_trace(
                go.Scatter(
                    x=days,
                    y=paths[index],
                    mode="lines",
                    line={"color": "rgba(100, 116, 139, 0.16)", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    figure.add_trace(
        go.Scatter(
            x=days,
            y=p95,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=days,
            y=p05,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.14)",
            name="5th–95th percentile",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=days,
            y=p75,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=days,
            y=p25,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.24)",
            name="25th–75th percentile",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=days,
            y=p50,
            mode="lines",
            line={"color": "#1d4ed8", "width": 3},
            name="Median",
        )
    )

    figure.update_layout(
        title=f"{ticker.upper()} historical-bootstrap Monte Carlo fan chart",
        xaxis_title="Trading day",
        yaxis_title="Simulated adjusted price",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return figure


def save_fan_chart(
    price_paths: ArrayLike,
    ticker: str,
    output_path: Path,
    sample_paths: int = 25,
) -> Path:
    """Save an interactive fan chart as a self-contained HTML file."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_fan_chart(price_paths, ticker, sample_paths)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    return output
