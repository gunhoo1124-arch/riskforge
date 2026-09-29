import numpy as np
import pytest

from riskforge.metrics import (
    expected_shortfall,
    maximum_drawdowns,
    summarize_risk,
    terminal_returns,
    value_at_risk,
)


def test_value_at_risk_and_expected_shortfall() -> None:
    losses = np.arange(1.0, 101.0)

    var = value_at_risk(losses, 0.95)
    es = expected_shortfall(losses, 0.95)

    assert var == pytest.approx(95.05)
    assert es == pytest.approx(98.0)


def test_terminal_returns() -> None:
    paths = np.array([[100.0, 105.0, 110.0], [100.0, 95.0, 80.0]])

    np.testing.assert_allclose(terminal_returns(paths), [0.10, -0.20])


def test_maximum_drawdowns() -> None:
    paths = np.array(
        [
            [100.0, 120.0, 90.0, 108.0],
            [100.0, 110.0, 121.0, 133.1],
        ]
    )

    np.testing.assert_allclose(maximum_drawdowns(paths), [0.25, 0.0])


def test_summarize_risk_probabilities() -> None:
    paths = np.array(
        [
            [100.0, 120.0],
            [100.0, 100.0],
            [100.0, 95.0],
            [100.0, 80.0],
        ]
    )

    summary = summarize_risk(paths, loss_threshold=0.10)

    assert summary.probability_of_loss == pytest.approx(0.50)
    assert summary.probability_exceeding_threshold == pytest.approx(0.25)
