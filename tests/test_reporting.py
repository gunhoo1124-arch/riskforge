import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from riskforge.metrics import RiskSummary
from riskforge.portfolio import analyze_portfolio_risk
from riskforge.quality import assess_price_quality
from riskforge.reporting import (
    analysis_fingerprint,
    audit_html,
    audit_json,
    create_portfolio_audit,
    create_single_asset_audit,
)


def _summary() -> RiskSummary:
    return RiskSummary(
        var_95=0.10,
        var_99=0.15,
        expected_shortfall_95=0.13,
        expected_shortfall_99=0.18,
        probability_of_loss=0.45,
        probability_exceeding_threshold=0.08,
        mean_maximum_drawdown=0.09,
        maximum_drawdown_95=0.17,
    )


def _single_result(ticker: str = "SPY") -> dict[str, object]:
    prices = pd.Series(
        np.linspace(100.0, 110.0, 81),
        index=pd.bdate_range("2026-06-08", periods=81),
        name=ticker,
    )
    return {
        "ticker": ticker,
        "prices": prices,
        "summary": _summary(),
        "horizon": 20,
        "paths": 10_000,
        "lookback": 80,
        "seed": 42,
        "loss_threshold": 0.10,
        "method": "moving-block",
        "block_size": 5,
        "investment_amount": 10_000,
    }


def test_fingerprint_is_order_independent_and_input_sensitive() -> None:
    first = analysis_fingerprint({"seed": 42, "ticker": "SPY"})
    reordered = analysis_fingerprint({"ticker": "SPY", "seed": 42})
    changed = analysis_fingerprint({"ticker": "SPY", "seed": 43})

    assert first == reordered
    assert first != changed


def test_single_audit_is_compact_reproducible_and_valid_json() -> None:
    result = _single_result()
    prices = result["prices"]
    assert isinstance(prices, pd.Series)
    quality = assess_price_quality(
        prices,
        minimum_returns=80,
        as_of=prices.index[-1],
    )

    first = create_single_asset_audit(
        result,
        quality=quality,
        generated_at=datetime(2026, 9, 28, tzinfo=UTC),
    )
    second = create_single_asset_audit(
        result,
        quality=quality,
        generated_at=datetime(2026, 9, 29, tzinfo=UTC),
    )
    decoded = json.loads(audit_json(first))

    assert first["fingerprint"] == second["fingerprint"]
    assert decoded["inputs"]["seed"] == 42
    assert "price_paths" not in audit_json(first)
    assert len(first["fingerprint"]) == 16


def test_audit_html_escapes_untrusted_labels() -> None:
    snapshot = create_single_asset_audit(_single_result("<script>alert(1)</script>"))

    rendered = audit_html(snapshot)

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;SCRIPT&gt;" in rendered


def test_portfolio_audit_records_seed_quality_and_summary_not_paths() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0002, 0.01, size=(90, 2))
    prices = pd.DataFrame(
        100 * np.exp(np.vstack((np.zeros(2), np.cumsum(returns, axis=0)))),
        index=pd.bdate_range("2026-05-25", periods=91),
        columns=["SPY", "TLT"],
    )
    result = analyze_portfolio_risk(prices, [0.6, 0.4], horizon=5, paths=200, seed=7)

    snapshot = create_portfolio_audit(result)
    serialized = audit_json(snapshot)

    assert snapshot["inputs"]["seed"] == 7
    assert snapshot["data"]["quality"]["overall_status"] in {"pass", "warning"}
    assert "portfolio_paths" not in serialized
    assert "risk_metrics" in snapshot
