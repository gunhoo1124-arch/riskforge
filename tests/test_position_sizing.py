import pytest

from riskforge.metrics import RiskSummary
from riskforge.position_sizing import (
    RiskMeasure,
    calculate_position_limit,
    normalize_risk_measure,
    risk_rate_from_summary,
)


@pytest.fixture
def summary() -> RiskSummary:
    return RiskSummary(
        var_95=0.08,
        var_99=0.12,
        expected_shortfall_95=0.11,
        expected_shortfall_99=0.16,
        probability_of_loss=0.45,
        probability_exceeding_threshold=0.05,
        mean_maximum_drawdown=0.09,
        maximum_drawdown_95=0.18,
    )


def test_risk_rate_from_summary_selects_requested_measure(summary: RiskSummary) -> None:
    assert risk_rate_from_summary(summary, RiskMeasure.EXPECTED_SHORTFALL_95) == 0.11
    assert risk_rate_from_summary(summary, "99% VaR") == 0.12
    assert risk_rate_from_summary(summary, RiskMeasure.MAXIMUM_DRAWDOWN_95) == 0.18


def test_risk_budget_can_bind_position_limit() -> None:
    result = calculate_position_limit(
        portfolio_value=100_000,
        current_position=12_000,
        current_price=100,
        risk_rate=0.20,
        risk_budget_fraction=0.02,
        concentration_cap_fraction=0.25,
    )

    assert result.risk_budget_dollars == pytest.approx(2_000)
    assert result.risk_budget_position_limit == pytest.approx(10_000)
    assert result.position_limit == pytest.approx(10_000)
    assert result.position_limit_fraction == pytest.approx(0.10)
    assert result.units_at_limit == pytest.approx(100)
    assert result.modeled_loss_at_limit == pytest.approx(2_000)
    assert result.current_modeled_loss == pytest.approx(2_400)
    assert result.amount_above_limit == pytest.approx(2_000)
    assert result.binding_constraint == "risk_budget"
    assert result.current_status == "above_limits"


def test_concentration_cap_can_bind_position_limit() -> None:
    result = calculate_position_limit(
        portfolio_value=100_000,
        current_position=8_000,
        current_price=50,
        risk_rate=0.05,
        risk_budget_fraction=0.02,
        concentration_cap_fraction=0.10,
    )

    assert result.risk_budget_position_limit == pytest.approx(40_000)
    assert result.concentration_position_limit == pytest.approx(10_000)
    assert result.position_limit == pytest.approx(10_000)
    assert result.binding_constraint == "concentration_cap"
    assert result.current_status == "within_limits"


def test_zero_current_position_is_reported_separately() -> None:
    result = calculate_position_limit(
        portfolio_value=50_000,
        current_position=0,
        current_price=250,
        risk_rate=0.10,
        risk_budget_fraction=0.02,
        concentration_cap_fraction=0.20,
    )

    assert result.current_status == "no_position"
    assert result.current_modeled_loss == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("portfolio_value", 0, "Portfolio value"),
        ("current_position", -1, "Current position"),
        ("current_price", 0, "Current price"),
        ("risk_rate", 0, "selected simulation measure"),
        ("risk_budget_fraction", 1.01, "Risk budget"),
        ("concentration_cap_fraction", 0, "Concentration cap"),
    ],
)
def test_position_limit_rejects_invalid_inputs(
    field: str,
    value: float,
    message: str,
) -> None:
    inputs = {
        "portfolio_value": 100_000,
        "current_position": 10_000,
        "current_price": 100,
        "risk_rate": 0.10,
        "risk_budget_fraction": 0.02,
        "concentration_cap_fraction": 0.20,
    }
    inputs[field] = value

    with pytest.raises(ValueError, match=message):
        calculate_position_limit(**inputs)


def test_unknown_risk_measure_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown risk measure"):
        normalize_risk_measure("Sharpe ratio")
