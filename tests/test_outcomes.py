import numpy as np
import pytest

from riskforge.outcomes import summarize_outcome_split


def test_outcome_split_separates_gains_losses_and_flat_paths() -> None:
    paths = np.array(
        [
            [100.0, 110.0],
            [100.0, 90.0],
            [100.0, 100.0],
            [100.0, 120.0],
        ]
    )

    result = summarize_outcome_split(paths)

    assert result.positive_paths == 2
    assert result.negative_paths == 1
    assert result.unchanged_paths == 1
    assert result.probability_positive == 0.5
    assert result.probability_negative == 0.25
    assert result.average_positive_return == pytest.approx(np.mean([0.10, 0.20]))
    assert result.average_negative_return == pytest.approx(-0.10)


def test_outcome_split_handles_no_positive_paths() -> None:
    result = summarize_outcome_split([[100.0, 95.0], [100.0, 90.0]])

    assert result.average_positive_return is None
    assert result.probability_positive == 0.0
