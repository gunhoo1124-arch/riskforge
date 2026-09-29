"""Risk-budget position limits for a single long investment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from riskforge.metrics import RiskSummary


class RiskMeasure(StrEnum):
    """Simulation statistics available as position-sizing yardsticks."""

    EXPECTED_SHORTFALL_95 = "95% Expected Shortfall"
    EXPECTED_SHORTFALL_99 = "99% Expected Shortfall"
    VAR_95 = "95% VaR"
    VAR_99 = "99% VaR"
    MAXIMUM_DRAWDOWN_95 = "95% maximum drawdown"


@dataclass(frozen=True, slots=True)
class PositionSizingResult:
    """A position limit implied by a modeled loss budget and concentration cap."""

    portfolio_value: float
    current_position: float
    current_price: float
    risk_rate: float
    risk_budget_fraction: float
    concentration_cap_fraction: float
    risk_budget_dollars: float
    risk_budget_position_limit: float
    concentration_position_limit: float
    position_limit: float
    position_limit_fraction: float
    units_at_limit: float
    modeled_loss_at_limit: float
    current_modeled_loss: float
    amount_above_limit: float
    binding_constraint: str
    current_status: str


def normalize_risk_measure(measure: RiskMeasure | str) -> RiskMeasure:
    """Return a validated risk-measure enum."""
    if isinstance(measure, RiskMeasure):
        return measure
    if not isinstance(measure, str):
        raise TypeError("Risk measure must be provided as text or a RiskMeasure value.")
    try:
        return RiskMeasure(measure)
    except ValueError as exc:
        choices = ", ".join(item.value for item in RiskMeasure)
        raise ValueError(f"Unknown risk measure. Choose one of: {choices}.") from exc


def risk_rate_from_summary(
    summary: RiskSummary,
    measure: RiskMeasure | str,
) -> float:
    """Extract the selected positive-loss rate from a simulation summary."""
    if not isinstance(summary, RiskSummary):
        raise TypeError("Summary must be a RiskSummary value.")
    selected = normalize_risk_measure(measure)
    values = {
        RiskMeasure.EXPECTED_SHORTFALL_95: summary.expected_shortfall_95,
        RiskMeasure.EXPECTED_SHORTFALL_99: summary.expected_shortfall_99,
        RiskMeasure.VAR_95: summary.var_95,
        RiskMeasure.VAR_99: summary.var_99,
        RiskMeasure.MAXIMUM_DRAWDOWN_95: summary.maximum_drawdown_95,
    }
    value = float(values[selected])
    if not np.isfinite(value):
        raise ValueError("The selected risk measure must be finite.")
    return value


def _finite_number(value: float, name: str) -> float:
    """Validate and normalize a real finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def calculate_position_limit(
    *,
    portfolio_value: float,
    current_position: float,
    current_price: float,
    risk_rate: float,
    risk_budget_fraction: float,
    concentration_cap_fraction: float,
) -> PositionSizingResult:
    """Calculate a long-only position limit under two user-defined guardrails.

    The risk-budget limit is ``portfolio risk budget / modeled loss rate``.
    The final limit is the smaller of that amount and the concentration cap.
    This is an arithmetic scenario limit, not an investment recommendation.
    """
    portfolio = _finite_number(portfolio_value, "Portfolio value")
    position = _finite_number(current_position, "Current position")
    price = _finite_number(current_price, "Current price")
    modeled_rate = _finite_number(risk_rate, "Modeled loss rate")
    budget_fraction = _finite_number(risk_budget_fraction, "Risk budget")
    concentration_fraction = _finite_number(
        concentration_cap_fraction,
        "Concentration cap",
    )

    if portfolio <= 0:
        raise ValueError("Portfolio value must be greater than zero.")
    if position < 0:
        raise ValueError("Current position cannot be negative.")
    if position > portfolio:
        raise ValueError(
            "Current position cannot exceed portfolio value in this long-only, no-leverage tool."
        )
    if price <= 0:
        raise ValueError("Current price must be greater than zero.")
    if not 0 < modeled_rate <= 1:
        raise ValueError(
            "The selected simulation measure must be a loss greater than 0% and no more "
            "than 100%. Choose a different risk yardstick if this run produced a gain cutoff."
        )
    if not 0 < budget_fraction <= 1:
        raise ValueError("Risk budget must be greater than 0% and no more than 100%.")
    if not 0 < concentration_fraction <= 1:
        raise ValueError("Concentration cap must be greater than 0% and no more than 100%.")

    risk_budget_dollars = portfolio * budget_fraction
    risk_limit = risk_budget_dollars / modeled_rate
    concentration_limit = portfolio * concentration_fraction
    position_limit = min(risk_limit, concentration_limit)

    if np.isclose(risk_limit, concentration_limit):
        binding_constraint = "both"
    elif risk_limit < concentration_limit:
        binding_constraint = "risk_budget"
    else:
        binding_constraint = "concentration_cap"

    if position == 0:
        current_status = "no_position"
    elif position <= position_limit or np.isclose(position, position_limit):
        current_status = "within_limits"
    else:
        current_status = "above_limits"

    return PositionSizingResult(
        portfolio_value=portfolio,
        current_position=position,
        current_price=price,
        risk_rate=modeled_rate,
        risk_budget_fraction=budget_fraction,
        concentration_cap_fraction=concentration_fraction,
        risk_budget_dollars=risk_budget_dollars,
        risk_budget_position_limit=risk_limit,
        concentration_position_limit=concentration_limit,
        position_limit=position_limit,
        position_limit_fraction=position_limit / portfolio,
        units_at_limit=position_limit / price,
        modeled_loss_at_limit=position_limit * modeled_rate,
        current_modeled_loss=position * modeled_rate,
        amount_above_limit=max(0.0, position - position_limit),
        binding_constraint=binding_constraint,
        current_status=current_status,
    )
