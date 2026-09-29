"""Plain-language positive and negative simulation outcome summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from riskforge.metrics import terminal_returns


@dataclass(frozen=True, slots=True)
class OutcomeSplit:
    """Conditional summary of simulated gains and losses."""

    positive_paths: int
    negative_paths: int
    unchanged_paths: int
    total_paths: int
    probability_positive: float
    probability_negative: float
    average_positive_return: float | None
    average_negative_return: float | None


def summarize_outcome_split(price_paths: ArrayLike) -> OutcomeSplit:
    """Separate terminal simulated returns into positive, negative, and flat groups."""
    returns = terminal_returns(price_paths)
    positive = returns > 0
    negative = returns < 0
    unchanged = ~(positive | negative)
    total = len(returns)
    return OutcomeSplit(
        positive_paths=int(positive.sum()),
        negative_paths=int(negative.sum()),
        unchanged_paths=int(unchanged.sum()),
        total_paths=total,
        probability_positive=float(positive.mean()),
        probability_negative=float(negative.mean()),
        average_positive_return=float(np.mean(returns[positive])) if positive.any() else None,
        average_negative_return=float(np.mean(returns[negative])) if negative.any() else None,
    )
