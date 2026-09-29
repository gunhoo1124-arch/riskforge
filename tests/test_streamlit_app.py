from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from riskforge.allocation import AllocationResult, optimize_allocation
from riskforge.backtesting import BacktestResult
from riskforge.factor_attribution import analyze_factor_exposures, apply_factor_stress
from riskforge.metrics import RiskSummary
from riskforge.ml_volatility import MlVolatilityResult, RidgeEvaluation
from riskforge.portfolio import PortfolioRiskResult, analyze_portfolio_risk
from riskforge.position_sizing import RiskMeasure, calculate_position_limit
from riskforge.robustness import RobustnessResult
from riskforge.volatility import analyze_volatility_model

APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"


def sample_simulation_result() -> dict[str, object]:
    historical_prices = np.linspace(100.0, 90.0, 100)
    return {
        "ticker": "SPY",
        "prices": pd.Series(
            historical_prices,
            index=pd.bdate_range("2026-01-01", periods=100),
            name="SPY",
        ),
        "log_returns": pd.Series(np.diff(np.log(historical_prices))),
        "price_paths": np.array(
            [
                [101.0, 103.0, 105.0],
                [101.0, 100.0, 98.0],
                [101.0, 95.0, 90.0],
            ]
        ),
        "summary": RiskSummary(
            var_95=0.10,
            var_99=0.12,
            expected_shortfall_95=0.15,
            expected_shortfall_99=0.18,
            probability_of_loss=0.42,
            probability_exceeding_threshold=0.08,
            mean_maximum_drawdown=0.07,
            maximum_drawdown_95=0.14,
        ),
        "horizon": 20,
        "paths": 10_000,
        "lookback": 1_260,
        "seed": 42,
        "loss_threshold": 0.10,
        "investment_amount": 10_000.0,
        "method": "moving-block",
        "block_size": 5,
    }


def sample_robustness_result() -> RobustnessResult:
    """Build a compact model-comparison result for the dashboard test."""
    model_comparison = pd.DataFrame(
        {
            "model": ["Independent days (IID)", "Moving blocks: 5 days"],
            "method": ["iid", "moving-block"],
            "block_size": [None, 5],
            "var_95": [0.08, 0.10],
            "var_99": [0.12, 0.14],
            "expected_shortfall_95": [0.11, 0.13],
            "probability_of_loss": [0.45, 0.47],
            "probability_exceeding_threshold": [0.04, 0.06],
            "maximum_drawdown_95": [0.15, 0.18],
        }
    )
    backtest_comparison = pd.DataFrame(
        {
            "model": ["Independent days (IID)", "Moving blocks: 5 days"],
            "observations": [50, 50],
            "violations": [3, 4],
            "expected_violations": [2.5, 2.5],
            "violation_rate": [0.06, 0.08],
            "kupiec_p_value": [0.76, 0.38],
            "independence_p_value": [0.82, 0.42],
            "conditional_coverage_p_value": [0.91, 0.55],
            "coverage_status": ["no_failure_detected", "limited_evidence"],
            "independence_status": ["no_failure_detected", "limited_evidence"],
            "longest_violation_run": [1, 2],
        }
    )
    lookback_comparison = pd.DataFrame(
        {
            "model": ["About 1 years (252 days)", "Full sample (756 days)"],
            "method": ["moving-block", "moving-block"],
            "block_size": [5, 5],
            "var_95": [0.09, 0.10],
            "var_99": [0.13, 0.14],
            "expected_shortfall_95": [0.12, 0.13],
            "probability_of_loss": [0.44, 0.47],
            "probability_exceeding_threshold": [0.05, 0.06],
            "maximum_drawdown_95": [0.16, 0.18],
            "lookback": [252, 756],
        }
    )
    return RobustnessResult(
        model_comparison=model_comparison,
        backtest_comparison=backtest_comparison,
        lookback_comparison=lookback_comparison,
        horizon=20,
        paths=1_000,
        seed=42,
        training_window=378,
        evaluation_returns=378,
    )


def sample_volatility_result():
    """Run a small deterministic adaptive-model comparison for the UI test."""
    prices = sample_simulation_result()["prices"]
    assert isinstance(prices, pd.Series)
    return analyze_volatility_model(
        prices,
        horizon=5,
        paths=100,
        training_window=60,
        seed=5,
    )


def sample_ml_result() -> MlVolatilityResult:
    """Build a compact nested-validation result for the ML dashboard test."""
    dates = pd.bdate_range("2026-01-05", periods=3)
    evaluation = RidgeEvaluation(
        predictions=pd.DataFrame(
            {
                "forecast_date": dates - pd.offsets.BDay(5),
                "outcome_date": dates,
                "actual_volatility": [0.18, 0.24, 0.16],
                "ml_predicted_volatility": [0.17, 0.21, 0.18],
                "ewma_predicted_volatility": [0.14, 0.19, 0.20],
                "ml_absolute_error": [0.01, 0.03, 0.02],
                "ewma_absolute_error": [0.04, 0.05, 0.04],
            }
        ),
        selected_alpha=1.0,
        initial_training_samples=30,
        testing_samples=3,
        ml_mae=0.02,
        ewma_mae=0.04,
        ml_rmse=0.022,
        ewma_rmse=0.044,
    )
    risk_comparison = pd.DataFrame(
        {
            "model": ["EWMA-only volatility", "Ridge ML + EWMA paths"],
            "var_95": [0.08, 0.10],
            "var_99": [0.12, 0.14],
            "expected_shortfall_95": [0.11, 0.13],
            "expected_shortfall_99": [0.15, 0.17],
            "probability_of_loss": [0.45, 0.47],
            "maximum_drawdown_95": [0.16, 0.18],
        }
    )
    paths = np.array(
        [
            [100, 101, 102, 103, 104, 105],
            [100, 99, 98, 97, 96, 95],
            [100, 102, 99, 101, 98, 100],
        ],
        dtype=float,
    )
    coefficients = pd.DataFrame(
        {
            "feature": ["log_volatility_5", "log_ewma_volatility"],
            "standardized_coefficient": [0.3, 0.2],
        }
    )
    current_features = pd.DataFrame(
        {
            "feature": ["log_volatility_5", "log_ewma_volatility"],
            "value": [-1.6, -1.7],
            "training_mean": [-1.8, -1.75],
        }
    )
    return MlVolatilityResult(
        evaluation=evaluation,
        current_forecast_annualized_volatility=0.22,
        current_ewma_annualized_volatility=0.19,
        current_risk_comparison=risk_comparison,
        ml_price_paths=paths,
        coefficients=coefficients,
        current_features=current_features,
        dataset_rows=45,
        horizon=5,
        paths=1_000,
        prediction_was_clipped=False,
    )


def sample_portfolio_result() -> PortfolioRiskResult:
    """Build a deterministic multi-asset result for the portfolio UI test."""
    rng = np.random.default_rng(17)
    daily_returns = rng.normal(
        loc=[0.0003, 0.0002, 0.0001],
        scale=[0.012, 0.009, 0.006],
        size=(180, 3),
    )
    prices = pd.DataFrame(
        100 * np.exp(np.vstack((np.zeros(3), np.cumsum(daily_returns, axis=0)))),
        index=pd.bdate_range("2025-01-02", periods=181),
        columns=["SPY", "QQQ", "TLT"],
    )
    return analyze_portfolio_risk(
        prices,
        [0.5, 0.3, 0.2],
        horizon=5,
        paths=200,
        initial_value=100_000,
        loss_threshold=0.10,
        seed=9,
    )


def sample_allocation_result(portfolio: PortfolioRiskResult) -> AllocationResult:
    """Build a small constrained allocation result for the dashboard test."""
    return optimize_allocation(
        portfolio.historical_prices,
        portfolio.weights,
        objective="minimum-volatility",
        horizon=5,
        paths=200,
        seed=7,
        max_weight=0.70,
        turnover_limit=0.25,
        expected_shortfall_limit=0.80,
        drawdown_limit=0.80,
        initial_value=portfolio.initial_value,
    )


def test_streamlit_app_loads_without_fetching_data() -> None:
    app = AppTest.from_file(APP_PATH)

    app.run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "RiskForge"
    assert app.text_input[0].label == "1. Enter any ticker"
    assert app.text_input[0].value == "SPY"
    assert app.segmented_control[0].label == "How should past market days be sampled?"
    assert app.segmented_control[0].value == "moving-block"
    assert app.info[0].value.startswith("For a useful first example")
    assert any(header.value == "Your first simulation, in three steps" for header in app.header)


def test_results_explain_risk_in_plain_language() -> None:
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_result"] = sample_simulation_result()

    app.run(timeout=15)

    assert not app.exception
    assert app.metric[0].label == "Chance of any loss"
    assert app.metric[0].value == "42 in 100"
    assert app.metric[1].label == "Worst 5-in-100 cutoff"
    assert any("42 out of 100" in markdown.value for markdown in app.markdown)
    assert any(
        "keeps the entire market sequence in its original order" in markdown.value
        for markdown in app.markdown
    )
    assert any("Moving-block bootstrap" in info.value for info in app.info)
    assert any(metric.label == "Worst actual outcome" for metric in app.metric)
    assert any(metric.label == "Current regime" for metric in app.metric)
    assert any(
        "market-state label is context, not a buy or sell signal" in warning.value
        for warning in app.warning
    )
    assert any("stress test, not a forecast" in warning.value for warning in app.warning)
    assert any(metric.label == "Audit fingerprint" for metric in app.metric)
    assert any(metric.label == "Data quality gate" for metric in app.metric)
    assert any("Bullish modeled outcomes" in markdown.value for markdown in app.markdown)
    assert any("Bearish modeled outcomes" in markdown.value for markdown in app.markdown)
    assert any("not a safe bet" in caption.value for caption in app.caption)
    assert any(subheader.value == "Export this analysis" for subheader in app.subheader)


def test_backtest_results_explain_calibration_without_overclaiming() -> None:
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_result"] = sample_simulation_result()
    outcome_dates = pd.bdate_range("2025-01-01", periods=10)
    forecasts = pd.DataFrame(
        {
            "forecast_date": outcome_dates - pd.offsets.BDay(1),
            "outcome_date": outcome_dates,
            "predicted_var": np.full(10, 0.03),
            "predicted_expected_shortfall": np.full(10, 0.05),
            "realized_return": np.array([-0.04, *([0.01] * 9)]),
            "realized_loss": np.array([0.04, *([-0.01] * 9)]),
            "violation": np.array([True, *([False] * 9)]),
        }
    )
    backtest = BacktestResult(
        forecasts=forecasts,
        confidence=0.95,
        observations=10,
        violations=1,
        expected_violations=0.5,
        observed_violation_rate=0.10,
        expected_violation_rate=0.05,
        violation_rate_ci_low=0.018,
        violation_rate_ci_high=0.404,
        kupiec_statistic=0.413,
        kupiec_p_value=0.52,
        status="limited_evidence",
        transition_00=8,
        transition_01=0,
        transition_10=1,
        transition_11=0,
        violation_after_safe_rate=0.0,
        violation_after_violation_rate=0.0,
        independence_statistic=0.0,
        independence_p_value=1.0,
        independence_status="limited_evidence",
        conditional_coverage_statistic=0.413,
        conditional_coverage_p_value=0.813,
        max_consecutive_violations=1,
    )
    app.session_state["riskforge_backtest"] = {
        "ticker": "SPY",
        "result": backtest,
        "horizon": "1 day",
        "training_window": "3 years",
        "testing_window": "3 years",
    }

    app.run(timeout=15)

    assert not app.exception
    assert any(metric.label == "Periods tested" and metric.value == "10" for metric in app.metric)
    assert any("too few expected violations" in warning.value for warning in app.warning)
    assert any("too few violations to judge" in warning.value for warning in app.warning)
    assert any("never proves" in caption.value for caption in app.caption)
    assert any(
        metric.label == "Longest run of violations" and metric.value == "1" for metric in app.metric
    )


def test_model_comparison_explains_assumption_risk() -> None:
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_result"] = sample_simulation_result()
    app.session_state["riskforge_robustness"] = {
        "ticker": "SPY",
        "result": sample_robustness_result(),
    }

    app.run(timeout=15)

    assert not app.exception
    assert any(metric.label == "Models compared" and metric.value == "2" for metric in app.metric)
    assert any(
        "does not automatically make one model correct" in warning.value for warning in app.warning
    )
    assert any(
        "model choice itself is an important source of uncertainty" in warning.value
        for warning in app.warning
    )
    assert any("does not remove sampling uncertainty" in caption.value for caption in app.caption)


def test_position_sizing_explains_guardrails_without_recommending_trade() -> None:
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_result"] = sample_simulation_result()
    app.session_state["riskforge_position"] = {
        "ticker": "SPY",
        "result": calculate_position_limit(
            portfolio_value=100_000,
            current_position=8_000,
            current_price=100,
            risk_rate=0.20,
            risk_budget_fraction=0.02,
            concentration_cap_fraction=0.25,
        ),
        "risk_measure": RiskMeasure.EXPECTED_SHORTFALL_95.value,
    }

    app.run(timeout=15)

    assert not app.exception
    assert any(
        metric.label == "Limit under chosen guardrails" and metric.value == "$10,000"
        for metric in app.metric
    )
    assert any("does not recommend a trade" in warning.value for warning in app.warning)
    assert any("does not make the position safe" in success.value for success in app.success)
    assert any(
        "Loss-budget limit" in markdown.value and "Final limit" in markdown.value
        for markdown in app.markdown
    )


def test_volatility_comparison_explains_out_of_sample_limits() -> None:
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_result"] = sample_simulation_result()
    app.session_state["riskforge_volatility"] = {
        "ticker": "SPY",
        "result": sample_volatility_result(),
    }

    app.run(timeout=15)

    assert not app.exception
    assert any(metric.label == "Current EWMA volatility" for metric in app.metric)
    assert any(
        "does not automatically mean more accurate" in warning.value for warning in app.warning
    )
    assert any(
        subheader.value == "3. Strict walk-forward comparison" for subheader in app.subheader
    )
    assert any("overfit this backtest" in caption.value for caption in app.caption)


def test_ml_lab_requires_out_of_sample_benchmark() -> None:
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_result"] = sample_simulation_result()
    app.session_state["riskforge_ml"] = {
        "ticker": "SPY",
        "result": sample_ml_result(),
    }

    app.run(timeout=15)

    assert not app.exception
    assert any(metric.label == "Current ML volatility forecast" for metric in app.metric)
    assert any(
        "Machine learning is not automatically smarter or safer" in warning.value
        for warning in app.warning
    )
    assert any(
        subheader.value == "1. Did ML forecast volatility better?" for subheader in app.subheader
    )
    assert any("not causal explanations" in caption.value for caption in app.caption)


def test_portfolio_lab_explains_dependence_and_limits() -> None:
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_portfolio"] = sample_portfolio_result()
    app.session_state["risk_lab_mode"] = "Portfolio"

    app.run(timeout=15)

    assert not app.exception
    assert any(header.value == "Portfolio risk lab" for header in app.header)
    assert any(metric.label == "Chance of portfolio loss" for metric in app.metric)
    assert any(metric.label == "Effective number of holdings" for metric in app.metric)
    assert any("Diversification can reduce some risk" in warning.value for warning in app.warning)
    assert any("correlations and volatility change" in info.value for info in app.info)
    assert any("does not maximize gains" in error.value for error in app.error)
    assert any(metric.label == "Audit fingerprint" for metric in app.metric)
    assert any(subheader.value == "1. Build your holdings cart" for subheader in app.subheader)
    assert any(subheader.value == "Export this portfolio analysis" for subheader in app.subheader)


def test_factor_lab_labels_linear_stress_and_unexplained_risk() -> None:
    portfolio = sample_portfolio_result()
    attribution = analyze_factor_exposures(
        portfolio.historical_prices,
        portfolio.historical_prices[["SPY", "TLT"]],
        portfolio.weights,
    )
    stress = apply_factor_stress(
        attribution,
        {"SPY": -0.20, "TLT": 0.08},
        initial_value=portfolio.initial_value,
    )
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_portfolio"] = portfolio
    app.session_state["riskforge_factor"] = {
        "portfolio_tickers": portfolio.tickers,
        "portfolio_weights": tuple(portfolio.weights.tolist()),
        "labels": {"SPY": "Broad U.S. equities", "TLT": "Long U.S. Treasuries"},
        "attribution": attribution,
        "stress": stress,
    }
    app.session_state["risk_lab_mode"] = "Portfolio"

    app.run(timeout=15)

    assert not app.exception
    assert any(metric.label == "Historical variation explained" for metric in app.metric)
    assert any(metric.label == "Factor-implied portfolio return" for metric in app.metric)
    assert any("Proxy beta is a historical association" in warning.value for warning in app.warning)
    assert any("linear sensitivity estimate" in error.value for error in app.error)
    assert any(subheader.value == "4. User-defined proxy shock" for subheader in app.subheader)


def test_allocation_sandbox_explains_constraints_and_estimation_risk() -> None:
    portfolio = sample_portfolio_result()
    allocation = sample_allocation_result(portfolio)
    app = AppTest.from_file(APP_PATH)
    app.session_state["riskforge_portfolio"] = portfolio
    app.session_state["riskforge_allocation"] = {
        "portfolio_tickers": portfolio.tickers,
        "portfolio_weights": tuple(portfolio.weights.tolist()),
        "result": allocation,
    }
    app.session_state["risk_lab_mode"] = "Portfolio"

    app.run(timeout=15)

    assert not app.exception
    assert any(metric.label == "One-way turnover" for metric in app.metric)
    assert any(
        subheader.value == "3. Current weights compared with the sandbox"
        for subheader in app.subheader
    )
    assert any(
        "An optimizer always returns the best answer" in warning.value for warning in app.warning
    )
    assert any("Do not interpret the sandbox weights" in error.value for error in app.error)
    assert allocation.constraint_status["passes"].all()
