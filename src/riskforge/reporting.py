"""Compact, reproducible audit exports for RiskForge analyses."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from riskforge import __version__
from riskforge.allocation import AllocationResult
from riskforge.factor_attribution import FactorAttributionResult, FactorStressResult
from riskforge.metrics import RiskSummary
from riskforge.portfolio import PortfolioRiskResult
from riskforge.quality import DataQualityReport, assess_price_quality

AUDIT_SCHEMA_VERSION = "1.0"
CORE_LIMITATIONS = [
    "Historical resampling cannot create shocks or structural changes absent from the sample.",
    "Model frequencies are conditional scenarios, not promised real-world probabilities.",
    "Taxes, fees, liquidity, execution, and the user's financial circumstances are omitted.",
    "The report is educational research output and is not investment advice.",
]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return [_json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported audit value: {type(value).__name__}.")


def analysis_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible analysis data."""
    canonical = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _price_digest(prices: pd.Series | pd.DataFrame) -> str:
    frame = prices.to_frame() if isinstance(prices, pd.Series) else prices
    serialized = frame.to_csv(index=True, date_format="%Y-%m-%d", float_format="%.17g")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _quality_payload(report: DataQualityReport) -> dict[str, Any]:
    return {
        "overall_status": report.overall_status.value,
        "observations": report.observations,
        "assets": report.assets,
        "start_date": report.start_date,
        "end_date": report.end_date,
        "checks": _json_safe(report.checks),
    }


def _risk_payload(summary: RiskSummary) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(summary).items()}


def _timestamp(value: datetime | None) -> str:
    instant = value or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _finish_snapshot(payload: dict[str, Any], generated_at: datetime | None) -> dict[str, Any]:
    safe_payload = _json_safe(payload)
    assert isinstance(safe_payload, dict)
    fingerprint = analysis_fingerprint(safe_payload)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "riskforge_version": __version__,
        "generated_at_utc": _timestamp(generated_at),
        "fingerprint": fingerprint,
        **safe_payload,
    }


def create_single_asset_audit(
    result: Mapping[str, Any],
    *,
    quality: DataQualityReport | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a compact audit snapshot without embedding full simulated paths."""
    prices = result.get("prices")
    summary = result.get("summary")
    if not isinstance(prices, pd.Series):
        raise TypeError("Single-asset audit requires a pandas price Series.")
    if not isinstance(summary, RiskSummary):
        raise TypeError("Single-asset audit requires a RiskSummary.")
    report = quality or assess_price_quality(
        prices,
        minimum_returns=max(2, min(len(prices) - 1, int(result.get("lookback", 60)))),
    )
    payload = {
        "analysis_type": "single_asset",
        "inputs": {
            "ticker": str(result.get("ticker", prices.name or "unknown")).upper(),
            "horizon_trading_days": int(result["horizon"]),
            "simulation_paths": int(result["paths"]),
            "lookback_returns": int(result["lookback"]),
            "seed": int(result["seed"]),
            "loss_threshold": float(result["loss_threshold"]),
            "bootstrap_method": str(result.get("method", "iid")),
            "block_size": int(result.get("block_size", 1)),
            "hypothetical_investment": float(result.get("investment_amount", 10_000)),
        },
        "data": {
            "source": "Yahoo Finance via yfinance",
            "price_field": "auto-adjusted close",
            "price_digest": _price_digest(prices),
            "latest_adjusted_price": float(prices.iloc[-1]),
            "quality": _quality_payload(report),
        },
        "risk_metrics": _risk_payload(summary),
        "limitations": CORE_LIMITATIONS,
    }
    return _finish_snapshot(payload, generated_at)


def create_portfolio_audit(
    result: PortfolioRiskResult,
    *,
    quality: DataQualityReport | None = None,
    factor_attribution: FactorAttributionResult | None = None,
    factor_stress: FactorStressResult | None = None,
    allocation: AllocationResult | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a portfolio audit snapshot with optional factor and allocation sections."""
    report = (
        quality
        or result.data_quality
        or assess_price_quality(
            result.historical_prices,
            minimum_returns=max(2, len(result.historical_prices) - 1),
        )
    )
    payload: dict[str, Any] = {
        "analysis_type": "portfolio",
        "inputs": {
            "tickers": list(result.tickers),
            "starting_weights": dict(zip(result.tickers, result.weights, strict=True)),
            "initial_value": result.initial_value,
            "horizon_trading_days": result.horizon,
            "simulation_paths": result.paths,
            "seed": result.seed,
            "loss_threshold": result.loss_threshold,
            "bootstrap_method": result.method.value,
            "block_size": result.block_size,
        },
        "data": {
            "source": "Yahoo Finance via yfinance",
            "price_field": "auto-adjusted close",
            "price_digest": _price_digest(result.historical_prices),
            "quality": _quality_payload(report),
        },
        "risk_metrics": _risk_payload(result.summary),
        "portfolio_diagnostics": {
            "effective_number_of_assets": result.effective_number_of_assets,
            "weighted_standalone_var_95": result.weighted_standalone_var_95,
            "weighted_standalone_es_95": result.weighted_standalone_es_95,
            "var_diversification_gap": result.var_diversification_gap,
            "es_diversification_gap": result.es_diversification_gap,
            "asset_risk": result.asset_risk,
            "tail_contributions": result.tail_contributions,
            "historical_correlation": result.correlation.reset_index().rename(
                columns={"index": "ticker"}
            ),
        },
        "limitations": CORE_LIMITATIONS
        + [
            "Cross-asset relationships can change sharply in stressed markets.",
            "Quoted prices are not converted to a common currency.",
        ],
    }

    if factor_attribution is not None:
        factor_payload: dict[str, Any] = {
            "factors": list(factor_attribution.factors),
            "observations": factor_attribution.observations,
            "portfolio_betas": factor_attribution.portfolio_betas,
            "portfolio_r_squared": factor_attribution.portfolio_r_squared,
            "portfolio_residual_annualized_volatility": (
                factor_attribution.portfolio_residual_annualized_volatility
            ),
            "condition_number": factor_attribution.condition_number,
        }
        if factor_stress is not None:
            factor_payload["stress"] = {
                "factor_shocks": factor_stress.factor_shocks,
                "portfolio_implied_return": factor_stress.portfolio_implied_return,
                "portfolio_implied_change": factor_stress.portfolio_implied_change,
            }
        payload["factor_analysis"] = factor_payload

    if allocation is not None:
        payload["allocation_sandbox"] = {
            "objective": allocation.objective.value,
            "optimized_weights": dict(
                zip(allocation.tickers, allocation.optimized_weights, strict=True)
            ),
            "turnover": allocation.turnover,
            "max_weight": allocation.max_weight,
            "turnover_limit": allocation.turnover_limit,
            "expected_shortfall_limit": allocation.expected_shortfall_limit,
            "drawdown_limit": allocation.drawdown_limit,
            "return_shrinkage": allocation.return_shrinkage,
            "seed": allocation.seed,
            "comparison": allocation.comparison,
            "constraint_status": allocation.constraint_status,
        }
    return _finish_snapshot(payload, generated_at)


def audit_json(snapshot: Mapping[str, Any]) -> str:
    """Serialize an audit snapshot as stable, human-readable JSON."""
    return json.dumps(
        _json_safe(snapshot),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def audit_html(snapshot: Mapping[str, Any]) -> str:
    """Render a standalone HTML audit report with escaped snapshot content."""
    safe = _json_safe(snapshot)
    assert isinstance(safe, dict)
    title = (
        "RiskForge portfolio audit"
        if safe.get("analysis_type") == "portfolio"
        else ("RiskForge single-asset audit")
    )
    fingerprint = html.escape(str(safe.get("fingerprint", "unavailable")))
    generated = html.escape(str(safe.get("generated_at_utc", "unavailable")))
    body = html.escape(audit_json(safe))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 1000px; margin: 2rem auto;
            padding: 0 1rem; color: #172033; }}
    .meta {{ background: #eef6ff; border-left: 4px solid #1677c8; padding: 1rem; }}
    pre {{ background: #111827; color: #e5eef8; padding: 1rem; overflow: auto;
           border-radius: 0.5rem; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta"><strong>Fingerprint:</strong> {fingerprint}<br>
  <strong>Generated (UTC):</strong> {generated}</div>
  <p>This compact report records inputs, data checks, results, and limitations. It excludes
  full simulated path arrays and is not investment advice.</p>
  <h2>Machine-readable snapshot</h2>
  <pre>{body}</pre>
</body>
</html>
"""
