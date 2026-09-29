"""Interactive Streamlit dashboard for RiskForge."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from riskforge.allocation import AllocationObjective, AllocationResult, optimize_allocation
from riskforge.backtesting import (
    BacktestResult,
    cumulative_coverage_frame,
    rolling_var_backtest,
)
from riskforge.data import (
    MarketDataError,
    download_adjusted_price_frame,
    download_adjusted_prices,
)
from riskforge.factor_attribution import (
    FactorAttributionResult,
    FactorStressResult,
    analyze_factor_exposures,
    apply_factor_stress,
)
from riskforge.metrics import RiskSummary, summarize_risk, terminal_returns
from riskforge.ml_volatility import MlVolatilityResult, analyze_ml_volatility
from riskforge.outcomes import OutcomeSplit, summarize_outcome_split
from riskforge.portfolio import (
    PortfolioRiskResult,
    analyze_portfolio_risk,
    investments_to_weights,
)
from riskforge.position_sizing import (
    PositionSizingResult,
    RiskMeasure,
    calculate_position_limit,
    risk_rate_from_summary,
)
from riskforge.quality import (
    DataQualityReport,
    QualityStatus,
    assess_price_quality,
    require_data_quality,
)
from riskforge.regimes import REGIME_ORDER, MarketRegime, RegimeAnalysis, analyze_market_regimes
from riskforge.reporting import (
    audit_html,
    audit_json,
    create_portfolio_audit,
    create_single_asset_audit,
)
from riskforge.returns import calculate_log_returns
from riskforge.robustness import RobustnessResult, analyze_model_robustness
from riskforge.simulation import BootstrapMethod, returns_to_price_paths, simulate_bootstrap
from riskforge.streamlit_components import holdings_cart
from riskforge.stress import (
    HistoricalStressResult,
    analyze_historical_stress,
    historical_exceedance_rate,
)
from riskforge.visualization import build_fan_chart
from riskforge.volatility import VolatilityAnalysisResult, analyze_volatility_model

HORIZONS = {
    "1 week": 5,
    "1 month": 20,
    "3 months": 63,
    "6 months": 126,
    "1 year": 252,
}
HISTORY_WINDOWS = {
    "1 year": 252,
    "3 years": 756,
    "5 years": 1_260,
    "10 years": 2_520,
}
BACKTEST_WINDOWS = {
    "1 year": 252,
    "3 years": 756,
    "5 years": 1_260,
}
REGIME_OUTLOOKS = {
    "1 week": 5,
    "1 month": 20,
    "3 months": 63,
}
REGIME_COLORS = {
    MarketRegime.CALM_ADVANCE: "#34D399",
    MarketRegime.CALM_DECLINE: "#FBBF24",
    MarketRegime.TURBULENT_ADVANCE: "#38BDF8",
    MarketRegime.TURBULENT_DECLINE: "#FB7185",
}
MODEL_LABELS = {
    BootstrapMethod.MOVING_BLOCK.value: "Consecutive-day blocks",
    BootstrapMethod.IID.value: "Independent days",
}
EWMA_DECAYS = {
    "Fast response (0.90)": 0.90,
    "Standard daily setting (0.94)": 0.94,
    "Slow response (0.97)": 0.97,
}
RISK_MEASURE_EXPLANATIONS = {
    RiskMeasure.EXPECTED_SHORTFALL_95: "Average modeled loss among the worst 5% of outcomes.",
    RiskMeasure.EXPECTED_SHORTFALL_99: "Average modeled loss among the worst 1% of outcomes.",
    RiskMeasure.VAR_95: "Loss cutoff exceeded by about 5% of modeled outcomes.",
    RiskMeasure.VAR_99: "Loss cutoff exceeded by about 1% of modeled outcomes.",
    RiskMeasure.MAXIMUM_DRAWDOWN_95: ("Peak-to-trough fall exceeded by about 5% of modeled paths."),
}
ALLOCATION_OBJECTIVE_LABELS = {
    AllocationObjective.MINIMUM_ES.value: "Lowest simulated tail loss",
    AllocationObjective.MINIMUM_VOLATILITY.value: "Lowest historical volatility",
    AllocationObjective.RISK_ADJUSTED.value: "Highest shrunk return per unit of volatility",
}

st.set_page_config(
    page_title="RiskForge | Monte Carlo risk lab",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "riskforge_result" not in st.session_state:
    st.session_state["riskforge_result"] = None
if "riskforge_backtest" not in st.session_state:
    st.session_state["riskforge_backtest"] = None
if "riskforge_robustness" not in st.session_state:
    st.session_state["riskforge_robustness"] = None
if "riskforge_position" not in st.session_state:
    st.session_state["riskforge_position"] = None
if "riskforge_volatility" not in st.session_state:
    st.session_state["riskforge_volatility"] = None
if "riskforge_ml" not in st.session_state:
    st.session_state["riskforge_ml"] = None
if "riskforge_portfolio" not in st.session_state:
    st.session_state["riskforge_portfolio"] = None
if "riskforge_factor" not in st.session_state:
    st.session_state["riskforge_factor"] = None
if "riskforge_allocation" not in st.session_state:
    st.session_state["riskforge_allocation"] = None

st.title("RiskForge", icon=":material/query_stats:")
st.markdown(
    "Stress-test an investment or portfolio by creating thousands of possible futures from "
    "past daily price moves. The goal is to understand a **range of risk**, not predict one "
    "price."
)
with st.container(horizontal=True):
    st.badge("Possible futures", icon=":material/route:", color="blue")
    st.badge("Downside risk", icon=":material/south_east:", color="orange")
    st.badge("Plain-language results", icon=":material/translate:", color="green")

analysis_mode = st.pills(
    "Choose a risk lab",
    options=["Single investment", "Portfolio"],
    default="Single investment",
    required=True,
    key="risk_lab_mode",
    help="Use the portfolio lab to model two to ten holdings together.",
)


@st.cache_data(ttl=3_600, max_entries=50, show_spinner=False)
def run_simulation(
    ticker: str,
    horizon: int,
    paths: int,
    lookback: int,
    seed: int,
    loss_threshold: float,
    method: str,
    block_size: int,
) -> dict[str, object]:
    """Run the complete simulation pipeline and cache identical requests."""
    prices = download_adjusted_prices(ticker, lookback)
    quality = assess_price_quality(prices, minimum_returns=lookback)
    require_data_quality(quality)
    log_returns = calculate_log_returns(prices)
    simulated_returns = simulate_bootstrap(
        log_returns.to_numpy(),
        horizon=horizon,
        paths=paths,
        method=method,
        block_size=block_size,
        seed=seed,
    )
    price_paths = returns_to_price_paths(simulated_returns, float(prices.iloc[-1]))
    summary = summarize_risk(price_paths, loss_threshold)
    return {
        "ticker": ticker.upper(),
        "prices": prices,
        "log_returns": log_returns,
        "price_paths": price_paths,
        "summary": summary,
        "horizon": horizon,
        "paths": paths,
        "lookback": lookback,
        "seed": seed,
        "loss_threshold": loss_threshold,
        "method": method,
        "block_size": block_size,
        "data_quality": quality,
    }


@st.cache_data(ttl=3_600, max_entries=20, show_spinner=False)
def run_backtest(
    ticker: str,
    training_window: int,
    testing_window: int,
    horizon: int,
    paths: int,
    confidence: float,
    seed: int,
    method: str,
    block_size: int,
) -> BacktestResult:
    """Download sufficient history and run a walk-forward VaR backtest."""
    prices = download_adjusted_prices(ticker, training_window + testing_window)
    log_returns = calculate_log_returns(prices)
    return rolling_var_backtest(
        log_returns,
        training_window=training_window,
        horizon=horizon,
        paths=paths,
        confidence=confidence,
        seed=seed,
        method=method,
        block_size=block_size,
    )


@st.cache_data(ttl=3_600, max_entries=20, show_spinner=False)
def run_robustness_analysis(
    prices: pd.Series,
    horizon: int,
    paths: int,
    seed: int,
    loss_threshold: float,
    selected_method: str,
    selected_block_size: int,
) -> RobustnessResult:
    """Compare model assumptions using the already-downloaded price history."""
    return analyze_model_robustness(
        prices,
        horizon=horizon,
        paths=paths,
        seed=seed,
        loss_threshold=loss_threshold,
        selected_method=selected_method,
        selected_block_size=selected_block_size,
    )


@st.cache_data(ttl=3_600, max_entries=20, show_spinner=False)
def run_volatility_analysis(
    prices: pd.Series,
    horizon: int,
    paths: int,
    training_window: int,
    seed: int,
    decay: float,
    loss_threshold: float,
    method: str,
    block_size: int,
) -> VolatilityAnalysisResult:
    """Run and cache the volatility-aware model and walk-forward comparison."""
    return analyze_volatility_model(
        prices,
        horizon=horizon,
        paths=paths,
        training_window=training_window,
        confidence=0.95,
        seed=seed,
        decay=decay,
        loss_threshold=loss_threshold,
        method=method,
        block_size=block_size,
    )


@st.cache_data(ttl=3_600, max_entries=10, show_spinner=False)
def run_ml_analysis(
    prices: pd.Series,
    horizon: int,
    paths: int,
    seed: int,
    loss_threshold: float,
    method: str,
    block_size: int,
) -> MlVolatilityResult:
    """Train, evaluate, and simulate the transparent ML volatility model."""
    return analyze_ml_volatility(
        prices,
        horizon=horizon,
        paths=paths,
        seed=seed,
        decay=0.94,
        loss_threshold=loss_threshold,
        method=method,
        block_size=block_size,
    )


@st.cache_data(ttl=3_600, max_entries=20, show_spinner=False)
def run_portfolio_simulation(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    horizon: int,
    paths: int,
    lookback: int,
    seed: int,
    loss_threshold: float,
    initial_value: float,
    method: str,
    block_size: int,
) -> PortfolioRiskResult:
    """Download aligned asset history and run the joint portfolio simulation."""
    prices = download_adjusted_price_frame(tickers, lookback)
    quality = assess_price_quality(prices, minimum_returns=lookback)
    require_data_quality(quality)
    result = analyze_portfolio_risk(
        prices,
        weights,
        horizon=horizon,
        paths=paths,
        initial_value=initial_value,
        loss_threshold=loss_threshold,
        method=method,
        block_size=block_size,
        seed=seed,
    )
    return replace(result, data_quality=quality)


@st.cache_data(ttl=3_600, max_entries=20, show_spinner=False)
def run_factor_attribution(
    asset_prices: pd.DataFrame,
    weights: tuple[float, ...],
    factors: tuple[str, ...],
    lookback: int,
) -> FactorAttributionResult:
    """Download shared proxy history and estimate portfolio factor exposures."""
    if len(factors) == 1:
        factor_prices = download_adjusted_prices(factors[0], lookback).to_frame()
    else:
        factor_prices = download_adjusted_price_frame(factors, lookback)
    return analyze_factor_exposures(asset_prices, factor_prices, weights)


@st.cache_data(ttl=3_600, max_entries=20, show_spinner=False)
def run_allocation_analysis(
    prices: pd.DataFrame,
    current_weights: tuple[float, ...],
    objective: str,
    horizon: int,
    paths: int,
    seed: int,
    method: str,
    block_size: int,
    max_weight: float,
    turnover_limit: float,
    expected_shortfall_limit: float | None,
    drawdown_limit: float | None,
    return_shrinkage: float,
    initial_value: float,
) -> AllocationResult:
    """Run and cache the deterministic constrained allocation sandbox."""
    return optimize_allocation(
        prices,
        current_weights,
        objective=objective,
        horizon=horizon,
        paths=paths,
        seed=seed,
        method=method,
        block_size=block_size,
        max_weight=max_weight,
        turnover_limit=turnover_limit,
        expected_shortfall_limit=expected_shortfall_limit,
        drawdown_limit=drawdown_limit,
        return_shrinkage=return_shrinkage,
        initial_value=initial_value,
    )


def percent(value: float, decimals: int = 1) -> str:
    """Format a decimal fraction as a percentage."""
    return f"{value:.{decimals}%}"


def out_of_100(probability: float) -> int:
    """Translate a probability into an approachable frequency."""
    return int(np.clip(round(probability * 100), 0, 100))


def describe_loss(loss: float, investment: float) -> str:
    """Describe a signed loss fraction in percentage and hypothetical dollars."""
    dollars = abs(loss * investment)
    if loss >= 0:
        return f"a {loss:.1%} loss (about ${dollars:,.0f})"
    return f"a {abs(loss):.1%} gain (about ${dollars:,.0f})"


def short_loss_label(loss: float) -> str:
    """Create a compact label that remains meaningful when VaR is negative."""
    if loss >= 0:
        return f"{loss:.1%} loss"
    return f"{abs(loss):.1%} gain"


def horizon_label(trading_days: int) -> str:
    """Return a familiar description of a trading-day horizon."""
    for label, days in HORIZONS.items():
        if days == trading_days:
            return label
    return f"{trading_days} trading days"


def bootstrap_model_label(
    method: BootstrapMethod,
    block_size: int,
    horizon: int,
) -> str:
    """Return a plain-language label for a configured bootstrap model."""
    if method is BootstrapMethod.IID:
        return "Independent-day bootstrap"
    effective_size = min(block_size, horizon)
    day_word = "day" if effective_size == 1 else "days"
    return f"Moving-block bootstrap ({effective_size} consecutive {day_word})"


def build_distribution_chart(price_paths: np.ndarray, ticker: str) -> go.Figure:
    """Build the terminal-return histogram and mark key risk quantiles."""
    returns = terminal_returns(price_paths)
    q01, q05, median = np.quantile(returns, [0.01, 0.05, 0.50])
    figure = go.Figure(
        go.Histogram(
            x=returns * 100,
            nbinsx=70,
            marker={"color": "#38BDF8", "line": {"color": "#07111F", "width": 0.4}},
            opacity=0.82,
            name="Simulated outcomes",
        )
    )
    for value, label, color in (
        (q01, "Worst 1-in-100 cutoff", "#FB7185"),
        (q05, "Worst 5-in-100 cutoff", "#FBBF24"),
        (median, "Middle outcome", "#A7F3D0"),
    ):
        figure.add_vline(
            x=value * 100,
            line_color=color,
            line_width=2,
            annotation_text=f"{label}: {value:.1%}",
            annotation_position="top",
        )
    figure.update_layout(
        title=f"Where {ticker}'s simulated outcomes landed",
        xaxis_title="Gain or loss at the end (%)",
        yaxis_title="Number of simulated futures",
        template="plotly_dark",
        bargap=0.03,
        showlegend=False,
    )
    return figure


def build_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    """Build a readable historical-correlation heatmap."""
    labels = [str(column) for column in correlation.columns]
    figure = go.Figure(
        go.Heatmap(
            z=correlation.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(correlation.to_numpy(), 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} with %{x}: %{z:.2f}<extra></extra>",
            colorbar={"title": "Correlation"},
        )
    )
    figure.update_layout(
        title="How the holdings moved together in the selected history",
        template="plotly_dark",
        height=max(420, 42 * len(labels)),
        xaxis_title="Holding",
        yaxis_title="Holding",
    )
    return figure


def build_history_chart(prices: pd.Series, ticker: str) -> go.Figure:
    """Build an adjusted-price history chart for the sampled window."""
    figure = go.Figure(
        go.Scatter(
            x=prices.index,
            y=prices.to_numpy(),
            mode="lines",
            line={"color": "#38BDF8", "width": 2},
            name="Adjusted closing price",
        )
    )
    figure.update_layout(
        title=f"Past {ticker} prices used to create the scenarios",
        xaxis_title=None,
        yaxis_title="Adjusted price",
        template="plotly_dark",
        showlegend=False,
    )
    return figure


def build_historical_stress_chart(
    stress: HistoricalStressResult,
    simulated_var_95: float,
    simulated_var_99: float,
    ticker: str,
) -> go.Figure:
    """Plot realized rolling losses against the simulation's loss cutoffs."""
    periods = stress.periods
    episodes = stress.worst_episodes
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=periods["end_date"],
            y=periods["loss"] * 100,
            mode="lines",
            line={"color": "#94A3B8", "width": 1.5},
            name="Actual rolling outcome",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=episodes["end_date"],
            y=episodes["loss"] * 100,
            mode="markers",
            marker={"color": "#FB7185", "size": 9, "symbol": "diamond"},
            name="Distinct worst episodes",
        )
    )
    figure.add_hline(
        y=simulated_var_95 * 100,
        line_color="#FBBF24",
        line_width=2,
        line_dash="dash",
        annotation_text="Simulated 95% VaR",
        annotation_position="top left",
    )
    figure.add_hline(
        y=simulated_var_99 * 100,
        line_color="#F97316",
        line_width=2,
        line_dash="dot",
        annotation_text="Simulated 99% VaR",
        annotation_position="bottom left",
    )
    figure.add_hline(y=0, line_color="rgba(148, 163, 184, 0.35)", line_width=1)
    figure.update_layout(
        title=f"Actual {ticker} outcomes over rolling {stress.horizon}-day periods",
        xaxis_title="Period ending date",
        yaxis_title="Loss (+) or gain (-), %",
        template="plotly_dark",
        hovermode="x unified",
    )
    return figure


def build_regime_chart(
    analysis: RegimeAnalysis,
    ticker: str,
) -> go.Figure:
    """Plot price regimes alongside rolling annualized volatility."""
    history = analysis.history
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
    )
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["price"],
            mode="lines",
            line={"color": "rgba(148, 163, 184, 0.55)", "width": 1.2},
            name="Adjusted price",
            hovertemplate="$%{y:,.2f}<extra>Adjusted price</extra>",
        ),
        row=1,
        col=1,
    )
    for regime in REGIME_ORDER:
        observations = history[history["regime"] == regime.value]
        if observations.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=observations["date"],
                y=observations["price"],
                mode="markers",
                marker={"color": REGIME_COLORS[regime], "size": 5, "opacity": 0.78},
                name=regime.value,
                hovertemplate=(
                    f"{regime.value}<br>$%{{y:,.2f}}<br>%{{x|%b %d, %Y}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["annualized_volatility"] * 100,
            mode="lines",
            line={"color": "#38BDF8", "width": 1.8},
            name="Rolling volatility",
            hovertemplate="%{y:.1f}%<extra>Annualized volatility</extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["volatility_threshold"] * 100,
            mode="lines",
            line={"color": "#FBBF24", "width": 1.5, "dash": "dash"},
            name="Calm/turbulent boundary",
            hovertemplate="%{y:.1f}%<extra>Expanding median volatility</extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        title=f"How {ticker}'s market state changed through the sample",
        template="plotly_dark",
        height=680,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06, "x": 0},
    )
    figure.update_yaxes(title_text="Adjusted price", row=1, col=1)
    figure.update_yaxes(title_text="Annualized volatility (%)", row=2, col=1)
    figure.update_xaxes(title_text="Date", row=2, col=1)
    return figure


def build_backtest_loss_chart(result: BacktestResult) -> go.Figure:
    """Compare each predicted loss cutoff with the subsequently realized loss."""
    forecasts = result.forecasts
    violations = forecasts[forecasts["violation"]]
    confidence_label = f"{result.confidence:.0%} VaR cutoff"
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=forecasts["outcome_date"],
            y=forecasts["realized_loss"] * 100,
            mode="lines+markers",
            line={"color": "#94A3B8", "width": 1.5},
            marker={"size": 5},
            name="Loss that actually occurred",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecasts["outcome_date"],
            y=forecasts["predicted_var"] * 100,
            mode="lines",
            line={"color": "#FBBF24", "width": 2},
            name=confidence_label,
        )
    )
    if not violations.empty:
        figure.add_trace(
            go.Scatter(
                x=violations["outcome_date"],
                y=violations["realized_loss"] * 100,
                mode="markers",
                marker={"color": "#FB7185", "size": 10, "symbol": "x"},
                name="Cutoff exceeded",
            )
        )
    figure.add_hline(y=0, line_color="rgba(148, 163, 184, 0.35)", line_width=1)
    figure.update_layout(
        title="Predicted loss cutoff compared with what happened next",
        xaxis_title=None,
        yaxis_title="Loss (+) or gain (-), %",
        template="plotly_dark",
        hovermode="x unified",
    )
    return figure


def build_volatility_backtest_chart(result: VolatilityAnalysisResult) -> go.Figure:
    """Compare fixed and volatility-aware VaR against identical realized outcomes."""
    baseline = result.baseline_backtest.forecasts
    adaptive = result.adaptive_backtest.forecasts
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=baseline["outcome_date"],
            y=baseline["realized_loss"] * 100,
            mode="lines+markers",
            line={"color": "#94A3B8", "width": 1.5},
            marker={"size": 5},
            name="Loss that occurred",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=baseline["outcome_date"],
            y=baseline["predicted_var"] * 100,
            mode="lines",
            line={"color": "#FBBF24", "width": 2},
            name="Fixed bootstrap 95% VaR",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=adaptive["outcome_date"],
            y=adaptive["predicted_var"] * 100,
            mode="lines",
            line={"color": "#38BDF8", "width": 2},
            name="EWMA-aware 95% VaR",
        )
    )
    figure.add_hline(y=0, line_color="rgba(148, 163, 184, 0.35)", line_width=1)
    figure.update_layout(
        title="Both loss cutoffs compared with the same later outcomes",
        xaxis_title=None,
        yaxis_title="Loss (+) or gain (-), %",
        template="plotly_dark",
        hovermode="x unified",
    )
    return figure


def build_cumulative_coverage_chart(result: BacktestResult) -> go.Figure:
    """Show whether VaR violations accumulate near their expected rate."""
    coverage = cumulative_coverage_frame(result)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=coverage["outcome_date"],
            y=coverage["upper_95_bound"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=coverage["outcome_date"],
            y=coverage["lower_95_bound"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.14)",
            name="95% expected range",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=coverage["outcome_date"],
            y=coverage["expected_violations"],
            mode="lines",
            line={"color": "#38BDF8", "width": 2, "dash": "dash"},
            name="Expected violations",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=coverage["outcome_date"],
            y=coverage["actual_violations"],
            mode="lines+markers",
            line={"color": "#FB7185", "width": 2},
            marker={"size": 5},
            name="Actual violations",
        )
    )
    figure.update_layout(
        title="How loss-cutoff violations accumulated",
        xaxis_title=None,
        yaxis_title="Cumulative violations",
        template="plotly_dark",
        hovermode="x unified",
    )
    return figure


def show_backtest_status(result: BacktestResult) -> None:
    """Explain the coverage result without presenting it as proof of validity."""
    if result.status == "limited_evidence":
        st.warning(
            "There are too few expected violations for a strong statistical conclusion. "
            "Use a longer testing period or a shorter forecast horizon.",
            icon=":material/hourglass:",
        )
    elif result.status == "risk_underestimated":
        st.error(
            "The model produced significantly more loss-cutoff violations than expected. "
            "It appears to have underestimated risk in this period.",
            icon=":material/error:",
        )
    elif result.status == "risk_overestimated":
        st.warning(
            "The model produced significantly fewer violations than expected. It may be "
            "overstating risk or using cutoffs that are too wide.",
            icon=":material/info:",
        )
    else:
        st.success(
            "This test did not detect a statistically significant coverage failure. That is "
            "encouraging, but it is not proof that the model is correct or future-safe.",
            icon=":material/check_circle:",
        )


def show_independence_status(result: BacktestResult) -> None:
    """Explain whether VaR violations show statistically detectable dependence."""
    if result.independence_status == "limited_evidence":
        st.warning(
            "There are too few violations to judge whether the model's misses cluster. "
            "That is uncertainty, not evidence that the misses are independent.",
            icon=":material/hourglass:",
        )
    elif result.independence_status == "clustered":
        st.error(
            "The model's misses clustered together more than random timing would suggest. "
            "Risk may have changed faster than this historical model could adapt.",
            icon=":material/crisis_alert:",
        )
    elif result.independence_status == "dependence_detected":
        st.warning(
            "The timing of the model's misses was not independent, although the detected "
            "pattern was not clustering. Investigate the forecast sequence before relying "
            "on the model.",
            icon=":material/warning:",
        )
    else:
        st.success(
            "This test did not detect violation clustering. That is encouraging, but a "
            "small sample can still miss a real pattern.",
            icon=":material/check_circle:",
        )


def format_optional_rate(rate: float | None) -> str:
    """Format a transition rate while preserving missing-data meaning."""
    return "Not enough data" if rate is None else f"{rate:.1%}"


def render_outcome_split(
    outcomes: OutcomeSplit,
    investment: float,
    *,
    subject: str,
) -> None:
    """Show conditional gain and loss outcomes with accessible bullish/bearish color cues."""
    positive_average = (
        "No paths finished higher"
        if outcomes.average_positive_return is None
        else (
            f"average gain {outcomes.average_positive_return:.1%} "
            f"(${outcomes.average_positive_return * investment:,.0f})"
        )
    )
    negative_average = (
        "No paths finished lower"
        if outcomes.average_negative_return is None
        else (
            f"average loss {abs(outcomes.average_negative_return):.1%} "
            f"(${abs(outcomes.average_negative_return * investment):,.0f})"
        )
    )
    st.subheader("Bullish and bearish modeled outcomes", icon=":material/candlestick_chart:")
    bullish, bearish = st.columns(2)
    with bullish.container(border=True, height="stretch"):
        st.markdown(":green[**:material/trending_up: Bullish modeled outcomes**]")
        st.markdown(
            f":green[**{out_of_100(outcomes.probability_positive)} in 100 paths for "
            f"{subject} finished higher; {positive_average}.**]"
        )
    with bearish.container(border=True, height="stretch"):
        st.markdown(":red[**:material/trending_down: Bearish modeled outcomes**]")
        st.markdown(
            f":red[**{out_of_100(outcomes.probability_negative)} in 100 paths for "
            f"{subject} finished lower; {negative_average}.**]"
        )
    flat_note = (
        f" {outcomes.unchanged_paths:,} simulated path(s) finished exactly unchanged."
        if outcomes.unchanged_paths
        else ""
    )
    st.caption(
        "Green means a positive simulated ending, not a safe bet or recommendation. Red "
        "means a negative simulated ending, not a sell signal." + flat_note
    )


def render_audit_panel(
    snapshot: dict[str, object],
    quality: DataQualityReport,
    *,
    file_stem: str,
) -> None:
    """Explain the quality gate and offer compact reproducibility downloads."""
    st.subheader("Reproducible audit pack", icon=":material/verified_user:")
    st.markdown(
        "This pack records the exact model settings, seed, data window, summarized results, "
        "and known limitations. The fingerprint changes when a material input or result "
        "changes, making two runs easier to compare."
    )

    status_labels = {
        QualityStatus.PASS: "Pass",
        QualityStatus.WARNING: "Review warnings",
        QualityStatus.FAIL: "Blocked",
    }
    with st.container(horizontal=True):
        st.metric("Audit fingerprint", str(snapshot["fingerprint"]), border=True)
        st.metric(
            "Data quality gate",
            status_labels[quality.overall_status],
            border=True,
        )
        st.metric("Price data through", quality.end_date or "Unavailable", border=True)

    if quality.overall_status is QualityStatus.PASS:
        st.success(
            "All automated price-integrity checks passed. This does not prove the data is "
            "economically correct or the model is suitable.",
            icon=":material/check_circle:",
        )
    elif quality.overall_status is QualityStatus.WARNING:
        st.warning(
            "The simulation ran, but at least one non-blocking data warning needs review.",
            icon=":material/warning:",
        )
    else:
        st.error(
            "The data failed a blocking integrity check. New simulations stop in this state.",
            icon=":material/error:",
        )

    quality_table = quality.checks.copy()
    quality_table["status"] = quality_table["status"].str.title()
    st.dataframe(
        quality_table,
        hide_index=True,
        width="stretch",
        column_config={
            "check": "Automated check",
            "status": "Status",
            "detail": "What RiskForge found",
        },
    )

    with st.container(horizontal=True):
        st.download_button(
            "Download audit JSON",
            data=audit_json(snapshot),
            file_name=f"{file_stem}_audit.json",
            mime="application/json",
            on_click="ignore",
            icon=":material/data_object:",
        )
        st.download_button(
            "Download readable audit report",
            data=audit_html(snapshot),
            file_name=f"{file_stem}_audit.html",
            mime="text/html",
            on_click="ignore",
            icon=":material/download:",
        )
    st.caption(
        "The audit pack excludes full simulated path arrays. Keep the JSON with research "
        "notes or screenshots so assumptions are not separated from the reported numbers."
    )


if analysis_mode == "Portfolio":
    st.header("Portfolio risk lab", icon=":material/donut_large:")
    st.markdown(
        "Model several holdings **together**. RiskForge keeps each historical market day's "
        "returns aligned across the holdings, then asks how your starting allocation might "
        "behave across thousands of joint scenarios."
    )
    st.warning(
        "Diversification can reduce some risk, but it cannot make a portfolio safe. Holdings "
        "that usually move differently can still fall together during a crisis.",
        icon=":material/warning:",
    )

    starter_portfolio = [
        {"ticker": "SPY", "amount": 50_000.0},
        {"ticker": "QQQ", "amount": 30_000.0},
        {"ticker": "TLT", "amount": 20_000.0},
    ]
    st.subheader("1. Build your holdings cart", icon=":material/shopping_cart:")
    st.markdown(
        "Add two to ten Yahoo Finance tickers and enter the dollars you would be willing to "
        "allocate to each one. Drag holdings to reorder them; RiskForge converts the dollar "
        "amounts into starting weights automatically."
    )
    cart_holdings = holdings_cart(
        starter_portfolio,
        key="portfolio_holdings_cart",
        max_holdings=10,
    )
    cart_preview = pd.DataFrame(cart_holdings, columns=["ticker", "amount"])
    preview_amounts = pd.to_numeric(cart_preview["amount"], errors="coerce")
    preview_total = float(preview_amounts.sum()) if preview_amounts.notna().any() else 0.0
    active_portfolio = st.session_state.get("riskforge_portfolio")
    if isinstance(active_portfolio, PortfolioRiskResult):
        preview_tickers = tuple(
            cart_preview["ticker"].fillna("").astype(str).str.strip().str.upper().tolist()
        )
        expected_amounts = active_portfolio.weights * active_portfolio.initial_value
        cart_matches_result = (
            preview_tickers == active_portfolio.tickers
            and len(preview_amounts) == len(expected_amounts)
            and preview_amounts.notna().all()
            and np.allclose(preview_amounts.to_numpy(), expected_amounts)
        )
        if not cart_matches_result:
            st.session_state["riskforge_portfolio"] = None
            st.session_state["riskforge_factor"] = None
            st.session_state["riskforge_allocation"] = None
            st.info(
                "The holdings cart changed, so the previous portfolio result was cleared. "
                "Analyze the updated cart to create matching results.",
                icon=":material/refresh:",
            )
    if preview_total > 0:
        preview_table = cart_preview.copy()
        preview_table["amount"] = preview_amounts
        preview_table["weight"] = preview_amounts / preview_total
        st.dataframe(
            preview_table,
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": "Ticker",
                "amount": st.column_config.NumberColumn(
                    "Amount to invest",
                    format="$%.2f",
                ),
                "weight": st.column_config.ProgressColumn(
                    "Implied starting weight",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
            },
        )
        st.caption(
            f"Current cart total: **${preview_total:,.0f}**. The cart is an allocation "
            "scenario, not a recommendation to invest this amount."
        )

    with st.form("portfolio_controls"):
        st.subheader("2. Describe the scenario")
        portfolio_columns = st.columns(3)
        with portfolio_columns[0]:
            portfolio_horizon_choice = st.selectbox(
                "Holding period",
                options=list(HORIZONS),
                index=1,
                key="portfolio_horizon",
            )
        with portfolio_columns[1]:
            portfolio_history_choice = st.selectbox(
                "Shared price history",
                options=list(HISTORY_WINDOWS),
                index=2,
                key="portfolio_history",
                help="Only dates with valid prices for every holding are retained.",
            )
            portfolio_loss_threshold = st.slider(
                "Loss level to watch (%)",
                min_value=0,
                max_value=50,
                value=10,
                step=1,
                key="portfolio_loss_threshold",
            )
        with portfolio_columns[2]:
            portfolio_paths_input = st.select_slider(
                "Number of joint futures",
                options=[1_000, 5_000, 10_000, 25_000, 50_000],
                value=10_000,
                key="portfolio_paths",
            )
            portfolio_seed_input = st.number_input(
                "Reproducibility number",
                min_value=0,
                value=42,
                step=1,
                key="portfolio_seed",
            )

        with st.expander("Advanced sampling settings", icon=":material/settings:"):
            portfolio_model_choice = st.segmented_control(
                "How should shared market days be sampled?",
                options=list(MODEL_LABELS),
                default=BootstrapMethod.MOVING_BLOCK.value,
                required=True,
                format_func=lambda option: MODEL_LABELS[str(option)],
                key="portfolio_sampling_model",
                width="stretch",
            )
            portfolio_block_size = st.select_slider(
                "Consecutive days kept together",
                options=[2, 3, 5, 10, 20],
                value=5,
                key="portfolio_block_size",
                help="Ignored for the independent-day baseline.",
            )

        portfolio_submitted = st.form_submit_button(
            "Analyze this portfolio",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    st.caption(
        "Educational research tool only. It ignores taxes, fees, liquidity, and currency "
        "conversion, and it is not investment advice."
    )

    if portfolio_submitted:
        try:
            input_rows = pd.DataFrame(cart_holdings, columns=["ticker", "amount"])
            if not 2 <= len(input_rows) <= 10:
                raise ValueError("Add between two and ten holdings to the cart.")
            input_rows["ticker"] = (
                input_rows["ticker"].fillna("").astype(str).str.strip().str.upper()
            )
            if (input_rows["ticker"] == "").any():
                raise ValueError("Every cart item needs a ticker.")
            if input_rows["ticker"].duplicated().any():
                raise ValueError("Each ticker can appear only once in the holdings cart.")
            input_rows["amount"] = pd.to_numeric(input_rows["amount"], errors="coerce")

            portfolio_tickers = tuple(input_rows["ticker"].tolist())
            weight_values, portfolio_value = investments_to_weights(
                input_rows["amount"].to_numpy(),
                len(portfolio_tickers),
            )
            portfolio_weights = tuple(weight_values.tolist())
            with st.skeleton(height=180):
                portfolio_result = run_portfolio_simulation(
                    portfolio_tickers,
                    portfolio_weights,
                    HORIZONS[str(portfolio_horizon_choice)],
                    int(portfolio_paths_input),
                    HISTORY_WINDOWS[str(portfolio_history_choice)],
                    int(portfolio_seed_input),
                    portfolio_loss_threshold / 100,
                    portfolio_value,
                    str(portfolio_model_choice),
                    int(portfolio_block_size),
                )
                st.session_state["riskforge_portfolio"] = portfolio_result
                st.session_state["riskforge_factor"] = None
                st.session_state["riskforge_allocation"] = None
            st.toast("Portfolio analysis complete", icon=":material/check_circle:")
        except (MarketDataError, ValueError, TypeError) as exc:
            st.session_state["riskforge_portfolio"] = None
            st.error(str(exc), icon=":material/error:")
        except Exception as exc:
            st.session_state["riskforge_portfolio"] = None
            st.error(f"The portfolio analysis could not be completed: {exc}")

    portfolio_result = st.session_state.get("riskforge_portfolio")
    if portfolio_result is None:
        st.subheader("What this adds beyond a one-ticker simulation")
        portfolio_steps = st.columns(3, border=True)
        with portfolio_steps[0]:
            st.markdown("**Shared market days**")
            st.markdown(
                "If stocks and bonds moved together on a historical date, the simulation "
                "keeps those same-day returns together."
            )
        with portfolio_steps[1]:
            st.markdown("**Weights that drift**")
            st.markdown(
                "Your weights set the starting dollars. RiskForge does not rebalance, so "
                "stronger holdings become a larger share along a path."
            )
        with portfolio_steps[2]:
            st.markdown("**Who drives bad outcomes**")
            st.markdown(
                "Tail contribution shows how much each holding added or offset during the "
                "portfolio's worst simulated 5% of endings."
            )
        st.info(
            "The starter portfolio is only a demonstration, not a recommended allocation. "
            "Edit the cart amounts, then select **Analyze this portfolio**.",
            icon=":material/lightbulb:",
        )
        st.stop()

    assert isinstance(portfolio_result, PortfolioRiskResult)
    portfolio_quality = portfolio_result.data_quality or assess_price_quality(
        portfolio_result.historical_prices,
        minimum_returns=max(2, len(portfolio_result.historical_log_returns)),
    )
    portfolio_summary = portfolio_result.summary
    portfolio_terminal_returns = terminal_returns(portfolio_result.portfolio_paths)
    portfolio_outcomes = summarize_outcome_split(portfolio_result.portfolio_paths)
    portfolio_median_return = float(np.median(portfolio_terminal_returns))
    portfolio_loss_count = out_of_100(portfolio_summary.probability_of_loss)
    portfolio_threshold_count = out_of_100(portfolio_summary.probability_exceeding_threshold)
    portfolio_threshold = portfolio_result.loss_threshold
    portfolio_var_description = describe_loss(
        portfolio_summary.var_95,
        portfolio_result.initial_value,
    )
    portfolio_es_description = describe_loss(
        portfolio_summary.expected_shortfall_95,
        portfolio_result.initial_value,
    )

    st.subheader("Start here: what the portfolio simulation found")
    with st.container(border=True):
        st.markdown(
            f"""
            - **{portfolio_loss_count} out of 100** joint futures ended below the starting
              portfolio value.
            - The cutoff for the **worst 5 out of 100** was
              {portfolio_var_description}.
            - Those worst 5 out of 100 endings averaged
              {portfolio_es_description}.
            - **{portfolio_threshold_count} out of 100** futures lost at least
              **{portfolio_threshold:.0%}**.

            These are model frequencies from {portfolio_result.paths:,} simulated futures,
            not promised real-world odds.
            """
        )
        st.warning(
            "The bootstrap cannot invent a crash, correlation shift, delisting, or liquidity "
            "event absent from the chosen history. Treat the result as one stress lens, not a "
            "complete safety check.",
            icon=":material/warning:",
        )

    with st.container(horizontal=True):
        st.metric(
            "Chance of portfolio loss",
            f"{portfolio_loss_count} in 100",
            border=True,
        )
        st.metric(
            "Worst 5-in-100 cutoff",
            short_loss_label(portfolio_summary.var_95),
            border=True,
        )
        st.metric(
            "Average of worst 5 in 100",
            short_loss_label(portfolio_summary.expected_shortfall_95),
            border=True,
        )
        st.metric(
            "Middle simulated ending",
            f"{portfolio_median_return:+.1%}",
            help="The median is a middle scenario, not an expected or guaranteed return.",
            border=True,
        )

    render_outcome_split(
        portfolio_outcomes,
        portfolio_result.initial_value,
        subject="the portfolio",
    )

    (
        portfolio_overview,
        portfolio_holdings,
        portfolio_factors,
        portfolio_allocation,
        portfolio_audit,
        portfolio_limits,
    ) = st.tabs(
        [
            ":material/route: Portfolio futures",
            ":material/hub: Holdings and dependence",
            ":material/account_tree: Factor and stress lab",
            ":material/tune: Allocation sandbox",
            ":material/verified_user: Audit and export",
            ":material/menu_book: Assumptions and limits",
        ]
    )

    with portfolio_overview:
        st.markdown(
            "**How to read this:** every path is the value of all holdings combined. The "
            "starting weights are applied once; they drift as simulated prices change."
        )
        portfolio_fan_chart = build_fan_chart(
            portfolio_result.portfolio_paths,
            "Portfolio",
            sample_paths=20,
        )
        portfolio_fan_chart.update_layout(
            title=(
                "Joint buy-and-hold portfolio futures over "
                f"{horizon_label(portfolio_result.horizon)}"
            ),
            yaxis_title="Simulated portfolio value ($)",
            template="plotly_dark",
            height=600,
        )
        st.plotly_chart(
            portfolio_fan_chart,
            width="stretch",
            config={"displaylogo": False},
        )
        st.plotly_chart(
            build_distribution_chart(portfolio_result.portfolio_paths, "portfolio"),
            width="stretch",
            config={"displaylogo": False},
        )

        st.subheader("A cautious diversification comparison")
        st.markdown(
            "RiskForge compares the portfolio result with a simple weighted average of each "
            "holding's standalone risk under the same simulated scenarios. A positive gap "
            "means combining the holdings reduced this particular modeled loss estimate."
        )
        with st.container(horizontal=True):
            st.metric(
                "Portfolio 95% VaR",
                f"{portfolio_summary.var_95:.1%}",
                border=True,
            )
            st.metric(
                "Weighted standalone VaR",
                f"{portfolio_result.weighted_standalone_var_95:.1%}",
                border=True,
            )
            st.metric(
                "Modeled VaR gap",
                f"{portfolio_result.var_diversification_gap:+.1%}",
                help="Weighted standalone VaR minus portfolio VaR.",
                border=True,
            )
            st.metric(
                "Modeled ES gap",
                f"{portfolio_result.es_diversification_gap:+.1%}",
                help="Weighted standalone Expected Shortfall minus portfolio ES.",
                border=True,
            )
        if portfolio_result.var_diversification_gap <= 0:
            st.warning(
                "This run did not show a positive VaR diversification gap. Do not assume that "
                "adding more ticker symbols automatically reduces downside risk.",
                icon=":material/balance:",
            )
        else:
            st.info(
                "The modeled gap is conditional on this history and these weights. It can "
                "shrink or reverse when correlations and volatility change.",
                icon=":material/info:",
            )

    with portfolio_holdings:
        allocation_table = portfolio_result.asset_risk.copy()
        st.subheader("Starting allocation and standalone risk")
        st.dataframe(
            allocation_table,
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": "Ticker",
                "weight": st.column_config.NumberColumn("Starting weight", format="percent"),
                "starting_value": st.column_config.NumberColumn(
                    "Starting value",
                    format="dollar",
                ),
                "historical_volatility": st.column_config.NumberColumn(
                    "Historical annualized volatility",
                    format="percent",
                ),
                "standalone_var_95": st.column_config.NumberColumn(
                    "Standalone 95% VaR",
                    format="percent",
                ),
                "standalone_es_95": st.column_config.NumberColumn(
                    "Standalone worst-5% average",
                    format="percent",
                ),
            },
        )
        with st.container(horizontal=True):
            st.metric(
                "Largest starting weight",
                f"{portfolio_result.weights.max():.1%}",
                border=True,
            )
            st.metric(
                "Effective number of holdings",
                f"{portfolio_result.effective_number_of_assets:.2f}",
                help=(
                    "Inverse concentration: equal weights across three holdings equals 3; "
                    "uneven weights produce a smaller number."
                ),
                border=True,
            )
            st.metric(
                "Holdings modeled",
                f"{len(portfolio_result.tickers)}",
                border=True,
            )

        st.subheader("Historical co-movement")
        st.plotly_chart(
            build_correlation_heatmap(portfolio_result.correlation),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "Correlation describes linear same-day co-movement from -1 to +1. It does not "
            "prove causation and can change sharply in stressed markets."
        )

        st.subheader("Who contributed to the worst simulated endings?")
        tail_table = portfolio_result.tail_contributions.copy()
        contribution_chart = tail_table.set_index("ticker")[["es_contribution"]].rename(
            columns={"es_contribution": "Contribution to portfolio loss"}
        )
        st.bar_chart(
            contribution_chart,
            y="Contribution to portfolio loss",
            y_label="Fraction of starting portfolio value",
        )
        st.dataframe(
            tail_table,
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": "Ticker",
                "weight": st.column_config.NumberColumn("Starting weight", format="percent"),
                "es_contribution": st.column_config.NumberColumn(
                    "Contribution to worst-5% average",
                    format="percent",
                ),
                "es_contribution_dollars": st.column_config.NumberColumn(
                    "Hypothetical dollar contribution",
                    format="dollar",
                ),
                "share_of_es": st.column_config.NumberColumn(
                    "Share of total tail loss",
                    format="percent",
                ),
            },
        )
        st.caption(
            "Contributions average each holding's gain or loss only inside the portfolio's "
            "worst 5% of endings. A negative contribution means that holding offset some loss "
            "in those scenarios; contributions add to portfolio Expected Shortfall."
        )

        normalized_history = (
            portfolio_result.historical_prices / portfolio_result.historical_prices.iloc[0] * 100
        )
        st.subheader("Aligned historical prices, rebased to 100")
        st.line_chart(normalized_history, y_label="Growth of 100 in quoted-price units")

    with portfolio_factors:
        st.markdown(
            "**What this asks:** when your holdings moved in the past, how much of that "
            "movement was associated with transparent traded proxies? Then, if you specify "
            "shocks to those proxies, what would the fitted linear sensitivities imply?"
        )
        st.warning(
            "Proxy beta is a historical association, not a causal law. Large shocks, market "
            "regime changes, nonlinear instruments, and company-specific news can make the "
            "actual result very different.",
            icon=":material/warning:",
        )

        starter_factors = pd.DataFrame(
            {
                "Label": [
                    "Broad U.S. equities",
                    "Long U.S. Treasuries",
                    "Gold",
                ],
                "Ticker": ["SPY", "TLT", "GLD"],
                "Shock %": [-20.0, 8.0, 5.0],
            }
        )
        with st.form("factor_stress_controls"):
            st.subheader("1. Choose one to six factor proxies")
            st.caption(
                "The defaults are examples, not a recommended model. You can use broad "
                "markets, sector ETFs, commodity funds, currencies, or other Yahoo Finance "
                "tickers relevant to this portfolio."
            )
            edited_factors = st.data_editor(
                starter_factors,
                key="factor_proxy_editor",
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Label": st.column_config.TextColumn(
                        "Plain-language label",
                        required=True,
                    ),
                    "Ticker": st.column_config.TextColumn(
                        "Proxy ticker",
                        required=True,
                        help="Each proxy ticker must be unique.",
                    ),
                    "Shock %": st.column_config.NumberColumn(
                        "Scenario price shock (%)",
                        min_value=-100.0,
                        max_value=200.0,
                        step=1.0,
                        format="%.1f",
                        required=True,
                        help=(
                            "This is a price-return shock to the proxy, not an interest-rate "
                            "change in percentage points."
                        ),
                    ),
                },
            )
            st.caption(
                "Example: TLT +8% means an 8% price rise in the bond ETF proxy. It does not "
                "mean interest rates rise by eight percentage points."
            )
            factor_submitted = st.form_submit_button(
                "Estimate exposures and run stress",
                type="primary",
                icon=":material/account_tree:",
            )

        if factor_submitted:
            try:
                factor_rows = edited_factors.copy()
                factor_rows["Label"] = factor_rows["Label"].fillna("").astype(str).str.strip()
                factor_rows["Ticker"] = (
                    factor_rows["Ticker"].fillna("").astype(str).str.strip().str.upper()
                )
                factor_rows = factor_rows[factor_rows["Ticker"] != ""]
                factor_rows["Shock %"] = pd.to_numeric(
                    factor_rows["Shock %"],
                    errors="coerce",
                )
                if not 1 <= len(factor_rows) <= 6:
                    raise ValueError("Choose between one and six factor proxies.")
                if (factor_rows["Label"] == "").any():
                    raise ValueError("Every factor proxy needs a plain-language label.")
                if factor_rows["Ticker"].duplicated().any():
                    raise ValueError("Each factor proxy ticker must be unique.")
                if factor_rows["Shock %"].isna().any():
                    raise ValueError("Every factor proxy needs a numeric scenario shock.")

                factor_tickers = tuple(factor_rows["Ticker"].tolist())
                factor_labels = dict(zip(factor_tickers, factor_rows["Label"], strict=True))
                factor_shocks = dict(
                    zip(
                        factor_tickers,
                        (factor_rows["Shock %"] / 100).tolist(),
                        strict=True,
                    )
                )
                with st.skeleton(height=180):
                    attribution_result = run_factor_attribution(
                        portfolio_result.historical_prices,
                        tuple(portfolio_result.weights.tolist()),
                        factor_tickers,
                        len(portfolio_result.historical_log_returns),
                    )
                    stress_result = apply_factor_stress(
                        attribution_result,
                        factor_shocks,
                        initial_value=portfolio_result.initial_value,
                    )
                    st.session_state["riskforge_factor"] = {
                        "portfolio_tickers": portfolio_result.tickers,
                        "portfolio_weights": tuple(portfolio_result.weights.tolist()),
                        "labels": factor_labels,
                        "attribution": attribution_result,
                        "stress": stress_result,
                    }
                st.toast("Factor and stress analysis complete", icon=":material/check_circle:")
            except (MarketDataError, ValueError, TypeError) as exc:
                st.session_state["riskforge_factor"] = None
                st.error(str(exc), icon=":material/error:")
            except Exception as exc:
                st.session_state["riskforge_factor"] = None
                st.error(f"The factor analysis could not be completed: {exc}")

        factor_payload = st.session_state.get("riskforge_factor")
        portfolio_identity = (
            portfolio_result.tickers,
            tuple(portfolio_result.weights.tolist()),
        )
        payload_identity = (
            (
                factor_payload.get("portfolio_tickers"),
                factor_payload.get("portfolio_weights"),
            )
            if factor_payload
            else None
        )
        if not factor_payload or payload_identity != portfolio_identity:
            st.info(
                "Select **Estimate exposures and run stress** to fit the proxy model and "
                "apply the example shocks. Adjust the proxies and shocks to match risks you "
                "actually want to examine.",
                icon=":material/lightbulb:",
            )
        else:
            attribution = factor_payload["attribution"]
            factor_stress = factor_payload["stress"]
            factor_labels = factor_payload["labels"]
            assert isinstance(attribution, FactorAttributionResult)
            assert isinstance(factor_stress, FactorStressResult)

            explained_percent = attribution.portfolio_r_squared
            max_vif = float(attribution.factor_diagnostics["variance_inflation_factor"].max())
            factor_count = len(attribution.factors)
            if explained_percent < 0.50:
                st.warning(
                    f"These proxies explained {explained_percent:.1%} of the portfolio's "
                    "historical daily variation. Most movement remained outside this factor "
                    "set, so the stress estimate is especially incomplete.",
                    icon=":material/data_alert:",
                )
            else:
                st.info(
                    f"These proxies explained {explained_percent:.1%} of historical daily "
                    "variation. The remaining variation and all model instability still matter.",
                    icon=":material/info:",
                )
            if max_vif >= 10 or attribution.condition_number >= 30:
                st.warning(
                    "Some proxies are strongly redundant. Individual beta estimates may be "
                    "unstable even if their combined fitted return looks reasonable.",
                    icon=":material/warning:",
                )

            with st.container(horizontal=True):
                st.metric(
                    "Historical variation explained",
                    f"{attribution.portfolio_r_squared:.1%}",
                    help="Regression R-squared; a fit statistic, not forecast accuracy.",
                    border=True,
                )
                st.metric(
                    "Unexplained annualized volatility",
                    f"{attribution.portfolio_residual_annualized_volatility:.1%}",
                    help="Volatility of the portfolio regression residuals.",
                    border=True,
                )
                st.metric(
                    "Shared days fitted",
                    f"{attribution.observations:,}",
                    border=True,
                )
                st.metric(
                    "Proxy factors",
                    f"{factor_count}",
                    border=True,
                )

            st.subheader("2. Estimated historical sensitivities")
            st.markdown(
                "A beta near 1 means a 1% proxy move was associated with roughly a 1% move "
                "in the same direction, holding the other proxies fixed. Negative beta means "
                "the fitted association ran in the opposite direction."
            )
            beta_table = attribution.asset_betas.copy()
            beta_table.loc["Portfolio"] = attribution.portfolio_betas
            beta_table.index.name = "Holding"
            beta_display = beta_table.reset_index()
            beta_display = beta_display.rename(
                columns={
                    ticker: f"{ticker} — {factor_labels[ticker]}" for ticker in attribution.factors
                }
            )
            st.dataframe(
                beta_display,
                hide_index=True,
                width="stretch",
                column_config={
                    column: st.column_config.NumberColumn(column, format="%.3f")
                    for column in beta_display.columns
                    if column != "Holding"
                },
            )
            portfolio_beta_chart = pd.DataFrame(
                {
                    "Historical beta": attribution.portfolio_betas,
                }
            )
            portfolio_beta_chart.index = [
                f"{ticker} — {factor_labels[ticker]}" for ticker in attribution.factors
            ]
            st.bar_chart(
                portfolio_beta_chart,
                y="Historical beta",
                y_label="Portfolio beta",
            )
            st.caption(
                "Betas use daily log returns and an intercept over the shared sample. They "
                "describe the starting-weight portfolio and may change across windows."
            )

            st.subheader("3. Where fitted historical variance came from")
            variance_table = attribution.variance_attribution.copy()
            variance_table["driver"] = variance_table["driver"].map(
                lambda driver: (
                    f"{driver} — {factor_labels[driver]}" if driver in factor_labels else driver
                )
            )
            variance_chart = variance_table.set_index("driver")[["share_of_total_variance"]]
            variance_chart = variance_chart.rename(
                columns={"share_of_total_variance": "Share of historical variance"}
            )
            st.bar_chart(
                variance_chart,
                y="Share of historical variance",
                y_label="Share of variance",
            )
            st.dataframe(
                variance_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "driver": "Driver",
                    "annualized_variance_contribution": st.column_config.NumberColumn(
                        "Annualized variance contribution",
                        format="%.4f",
                    ),
                    "share_of_total_variance": st.column_config.NumberColumn(
                        "Share of historical variance",
                        format="percent",
                    ),
                    "kind": "Type",
                },
            )
            st.caption(
                "Correlated factors can have negative component contributions because one "
                "proxy may offset another in this fitted sample. The residual is the variation "
                "the selected proxies did not explain."
            )

            st.subheader("4. User-defined proxy shock")
            st.error(
                "This is a linear sensitivity estimate, not a forecast or a full stress test. "
                "It assumes historical betas remain fixed during the shock and assigns no "
                "extra company-specific or residual loss.",
                icon=":material/gpp_maybe:",
            )
            with st.container(horizontal=True):
                st.metric(
                    "Factor-implied portfolio return",
                    f"{factor_stress.portfolio_implied_return:+.1%}",
                    border=True,
                )
                st.metric(
                    "Factor-implied dollar change",
                    f"${factor_stress.portfolio_implied_change:+,.0f}",
                    border=True,
                )
                st.metric(
                    "Starting portfolio value",
                    f"${factor_stress.initial_value:,.0f}",
                    border=True,
                )

            stress_factor_table = factor_stress.factor_contributions.copy()
            stress_factor_table["label"] = stress_factor_table["factor"].map(factor_labels)
            st.markdown("**Contribution by shocked proxy**")
            st.dataframe(
                stress_factor_table[
                    [
                        "factor",
                        "label",
                        "shock",
                        "portfolio_beta",
                        "portfolio_return_contribution",
                        "portfolio_dollar_contribution",
                    ]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "factor": "Proxy ticker",
                    "label": "Meaning",
                    "shock": st.column_config.NumberColumn("Entered shock", format="percent"),
                    "portfolio_beta": st.column_config.NumberColumn(
                        "Historical beta",
                        format="%.3f",
                    ),
                    "portfolio_return_contribution": st.column_config.NumberColumn(
                        "Implied portfolio contribution",
                        format="percent",
                    ),
                    "portfolio_dollar_contribution": st.column_config.NumberColumn(
                        "Implied dollar contribution",
                        format="dollar",
                    ),
                },
            )

            holding_stress_table = factor_stress.holding_impacts.copy()
            holding_chart = holding_stress_table.set_index("ticker")[
                ["portfolio_return_contribution"]
            ].rename(columns={"portfolio_return_contribution": "Portfolio impact"})
            st.markdown("**Contribution by holding**")
            st.bar_chart(
                holding_chart,
                y="Portfolio impact",
                y_label="Contribution to portfolio return",
            )
            st.dataframe(
                holding_stress_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "ticker": "Holding",
                    "starting_weight": st.column_config.NumberColumn(
                        "Starting weight",
                        format="percent",
                    ),
                    "factor_implied_return": st.column_config.NumberColumn(
                        "Factor-implied holding return",
                        format="percent",
                    ),
                    "portfolio_return_contribution": st.column_config.NumberColumn(
                        "Portfolio return contribution",
                        format="percent",
                    ),
                    "portfolio_dollar_contribution": st.column_config.NumberColumn(
                        "Portfolio dollar contribution",
                        format="dollar",
                    ),
                },
            )

            with st.expander("Regression quality and proxy overlap", icon=":material/science:"):
                st.markdown(
                    "Variance inflation factor (VIF) measures how much a proxy overlaps with "
                    "the other proxies. Values above 10 are a common warning sign, not a hard "
                    "law. R-squared measures historical in-sample fit, not future accuracy."
                )
                factor_diagnostics = attribution.factor_diagnostics.copy()
                factor_diagnostics["label"] = factor_diagnostics["factor"].map(factor_labels)
                st.dataframe(
                    factor_diagnostics,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "factor": "Proxy ticker",
                        "label": "Meaning",
                        "annualized_volatility": st.column_config.NumberColumn(
                            "Historical annualized volatility",
                            format="percent",
                        ),
                        "variance_inflation_factor": st.column_config.NumberColumn(
                            "VIF",
                            format="%.2f",
                        ),
                    },
                )
                factor_correlation = attribution.factor_correlation.copy()
                factor_correlation.index.name = "Proxy"
                st.markdown("**Historical proxy-return correlation**")
                st.dataframe(
                    factor_correlation.reset_index(),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        factor: st.column_config.NumberColumn(factor, format="%.2f")
                        for factor in attribution.factors
                    },
                )
                st.dataframe(
                    attribution.regression_diagnostics,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "series": "Holding or portfolio",
                        "r_squared": st.column_config.NumberColumn(
                            "R-squared",
                            format="percent",
                        ),
                        "adjusted_r_squared": st.column_config.NumberColumn(
                            "Adjusted R-squared",
                            format="percent",
                        ),
                        "annualized_volatility": st.column_config.NumberColumn(
                            "Annualized volatility",
                            format="percent",
                        ),
                        "residual_annualized_volatility": st.column_config.NumberColumn(
                            "Unexplained annualized volatility",
                            format="percent",
                        ),
                    },
                )
                st.caption(
                    f"Standardized factor-matrix condition number: "
                    f"{attribution.condition_number:.2f}. Larger values indicate more "
                    "unstable separation of individual factor effects."
                )

    with portfolio_allocation:
        st.markdown(
            "**What this does:** search for a different long-only allocation under explicit "
            "concentration, turnover, tail-loss, and drawdown limits. Current and candidate "
            "weights are judged on the same deterministic scenario set."
        )
        st.warning(
            "An optimizer always returns the best answer to the model it was given—not the "
            "best future portfolio. Small changes in data, expected returns, or constraints "
            "can produce very different weights.",
            icon=":material/warning:",
        )

        asset_count = len(portfolio_result.tickers)
        minimum_position_cap = int(np.ceil(100 / asset_count))
        default_position_cap = max(60, minimum_position_cap)
        with st.form("allocation_controls"):
            st.subheader("1. Choose the sandbox objective")
            allocation_objective = st.selectbox(
                "What should the sandbox prioritize?",
                options=list(ALLOCATION_OBJECTIVE_LABELS),
                index=0,
                format_func=lambda value: ALLOCATION_OBJECTIVE_LABELS[str(value)],
                key="allocation_objective",
                help=(
                    "Tail loss uses simulated Expected Shortfall. Volatility uses historical "
                    "covariance. The risk-adjusted objective also uses uncertain return estimates."
                ),
            )

            st.subheader("2. Set guardrails")
            allocation_guardrails = st.columns(2)
            with allocation_guardrails[0]:
                allocation_max_weight = st.slider(
                    "Maximum in any one holding (%)",
                    min_value=minimum_position_cap,
                    max_value=100,
                    value=default_position_cap,
                    step=1,
                    help=(
                        "The minimum available value is the smallest cap that could still "
                        "allow all weights to add to 100%."
                    ),
                )
                allocation_turnover = st.slider(
                    "Maximum one-way turnover (%)",
                    min_value=0,
                    max_value=100,
                    value=30,
                    step=5,
                    help=(
                        "One-way turnover is half the sum of absolute weight changes. A 30% "
                        "limit lets the sandbox reallocate at most 30% of portfolio value."
                    ),
                )
                allocation_enforce_es = st.checkbox(
                    "Enforce a 95% Expected Shortfall cap",
                    value=True,
                )
                allocation_es_limit = st.slider(
                    "Maximum average loss in worst 5% (%)",
                    min_value=0,
                    max_value=100,
                    value=25,
                    step=1,
                    disabled=not allocation_enforce_es,
                )
            with allocation_guardrails[1]:
                allocation_enforce_drawdown = st.checkbox(
                    "Enforce a 95% drawdown cap",
                    value=True,
                )
                allocation_drawdown_limit = st.slider(
                    "Maximum 95th-percentile drawdown (%)",
                    min_value=0,
                    max_value=100,
                    value=35,
                    step=1,
                    disabled=not allocation_enforce_drawdown,
                )
                allocation_return_shrinkage = st.slider(
                    "Expected-return shrinkage (%)",
                    min_value=0,
                    max_value=100,
                    value=75,
                    step=5,
                    help=(
                        "Pulls each asset's historical mean toward the cross-asset average. "
                        "This mainly matters for the risk-adjusted objective."
                    ),
                )
                allocation_paths = st.select_slider(
                    "Scenario paths used by the sandbox",
                    options=[300, 500, 1_000, 2_000],
                    value=500,
                    help="More paths reduce simulation noise but make optimization slower.",
                )
                allocation_seed = st.number_input(
                    "Sandbox reproducibility number",
                    min_value=0,
                    value=42,
                    step=1,
                )

            st.caption(
                "Risk caps apply only to this model's scenarios. Passing them does not prove "
                "that future losses or drawdowns will stay below the selected levels."
            )
            allocation_submitted = st.form_submit_button(
                "Run constrained allocation sandbox",
                type="primary",
                icon=":material/tune:",
            )

        if allocation_submitted:
            try:
                with st.skeleton(height=180):
                    allocation_result = run_allocation_analysis(
                        portfolio_result.historical_prices,
                        tuple(portfolio_result.weights.tolist()),
                        str(allocation_objective),
                        portfolio_result.horizon,
                        int(allocation_paths),
                        int(allocation_seed),
                        portfolio_result.method.value,
                        portfolio_result.block_size,
                        allocation_max_weight / 100,
                        allocation_turnover / 100,
                        (allocation_es_limit / 100 if allocation_enforce_es else None),
                        (allocation_drawdown_limit / 100 if allocation_enforce_drawdown else None),
                        allocation_return_shrinkage / 100,
                        portfolio_result.initial_value,
                    )
                    st.session_state["riskforge_allocation"] = {
                        "portfolio_tickers": portfolio_result.tickers,
                        "portfolio_weights": tuple(portfolio_result.weights.tolist()),
                        "result": allocation_result,
                    }
                st.toast("Allocation sandbox complete", icon=":material/check_circle:")
            except (ValueError, TypeError) as exc:
                st.session_state["riskforge_allocation"] = None
                st.error(str(exc), icon=":material/error:")
            except Exception as exc:
                st.session_state["riskforge_allocation"] = None
                st.error(f"The allocation sandbox could not be completed: {exc}")

        allocation_payload = st.session_state.get("riskforge_allocation")
        allocation_identity = (
            (
                allocation_payload.get("portfolio_tickers"),
                allocation_payload.get("portfolio_weights"),
            )
            if allocation_payload
            else None
        )
        current_identity = (
            portfolio_result.tickers,
            tuple(portfolio_result.weights.tolist()),
        )
        if not allocation_payload or allocation_identity != current_identity:
            st.info(
                "Select **Run constrained allocation sandbox** to compare the current weights "
                "with one model-generated alternative. Start with the suggested guardrails, "
                "then change one assumption at a time.",
                icon=":material/lightbulb:",
            )
        else:
            allocation_result = allocation_payload["result"]
            assert isinstance(allocation_result, AllocationResult)
            current_allocation = allocation_result.comparison.iloc[0]
            sandbox_allocation = allocation_result.comparison.iloc[1]
            objective_label = ALLOCATION_OBJECTIVE_LABELS[allocation_result.objective.value]
            es_change = float(
                sandbox_allocation["expected_shortfall_95"]
                - current_allocation["expected_shortfall_95"]
            )
            drawdown_change = float(
                sandbox_allocation["maximum_drawdown_95"]
                - current_allocation["maximum_drawdown_95"]
            )

            st.subheader("3. Current weights compared with the sandbox")
            st.info(
                f"The solver prioritized **{objective_label.lower()}** and found a feasible "
                f"candidate after evaluating {allocation_result.scenario_paths:,} shared "
                "joint-bootstrap scenarios. This is a research comparison, not an instruction "
                "to trade.",
                icon=":material/experiment:",
            )
            with st.container(horizontal=True):
                st.metric(
                    "One-way turnover",
                    f"{allocation_result.turnover:.1%}",
                    help="Approximate share of portfolio value moved from sells into buys.",
                    border=True,
                )
                st.metric(
                    "Largest sandbox weight",
                    f"{allocation_result.optimized_weights.max():.1%}",
                    border=True,
                )
                st.metric(
                    "Sandbox worst-5% average",
                    f"{float(sandbox_allocation['expected_shortfall_95']):.1%}",
                    delta=f"{es_change:+.1%} vs current",
                    delta_color="inverse",
                    border=True,
                )
                st.metric(
                    "Sandbox drawdown cutoff",
                    f"{float(sandbox_allocation['maximum_drawdown_95']):.1%}",
                    delta=f"{drawdown_change:+.1%} vs current",
                    delta_color="inverse",
                    border=True,
                )

            weight_chart = allocation_result.weight_changes.set_index("ticker")[
                [
                    "current_weight",
                    "sandbox_weight",
                ]
            ].rename(
                columns={
                    "current_weight": "Current weight",
                    "sandbox_weight": "Sandbox weight",
                }
            )
            st.bar_chart(
                weight_chart,
                y=["Current weight", "Sandbox weight"],
                y_label="Portfolio weight",
            )
            st.dataframe(
                allocation_result.weight_changes,
                hide_index=True,
                width="stretch",
                column_config={
                    "ticker": "Ticker",
                    "current_weight": st.column_config.NumberColumn(
                        "Current weight",
                        format="percent",
                    ),
                    "sandbox_weight": st.column_config.NumberColumn(
                        "Sandbox weight",
                        format="percent",
                    ),
                    "weight_change": st.column_config.NumberColumn(
                        "Weight change",
                        format="percent",
                    ),
                    "hypothetical_dollar_change": st.column_config.NumberColumn(
                        "Hypothetical dollar change",
                        format="dollar",
                    ),
                },
            )

            st.subheader("4. Risk and return comparison on the shared model")
            st.dataframe(
                allocation_result.comparison,
                hide_index=True,
                width="stretch",
                column_config={
                    "allocation": "Allocation",
                    "annualized_expected_return": st.column_config.NumberColumn(
                        "Shrunk annualized return estimate",
                        format="percent",
                    ),
                    "annualized_volatility": st.column_config.NumberColumn(
                        "Historical annualized volatility",
                        format="percent",
                    ),
                    "risk_adjusted_score": st.column_config.NumberColumn(
                        "Return / volatility score",
                        format="%.3f",
                    ),
                    "var_95": st.column_config.NumberColumn(
                        "95% VaR",
                        format="percent",
                    ),
                    "expected_shortfall_95": st.column_config.NumberColumn(
                        "Average worst 5% loss",
                        format="percent",
                    ),
                    "probability_of_loss": st.column_config.NumberColumn(
                        "Chance of loss",
                        format="percent",
                    ),
                    "maximum_drawdown_95": st.column_config.NumberColumn(
                        "95% drawdown cutoff",
                        format="percent",
                    ),
                    "median_terminal_return": st.column_config.NumberColumn(
                        "Median ending return",
                        format="percent",
                    ),
                },
            )

            comparison_days = np.arange(allocation_result.current_paths.shape[1])
            path_comparison = pd.DataFrame(
                {
                    "Trading day": comparison_days,
                    "Current median": np.median(
                        allocation_result.current_paths,
                        axis=0,
                    ),
                    "Sandbox median": np.median(
                        allocation_result.optimized_paths,
                        axis=0,
                    ),
                    "Current 5th percentile": np.quantile(
                        allocation_result.current_paths,
                        0.05,
                        axis=0,
                    ),
                    "Sandbox 5th percentile": np.quantile(
                        allocation_result.optimized_paths,
                        0.05,
                        axis=0,
                    ),
                }
            )
            st.line_chart(
                path_comparison,
                x="Trading day",
                y=[
                    "Current median",
                    "Sandbox median",
                    "Current 5th percentile",
                    "Sandbox 5th percentile",
                ],
                y_label="Portfolio value relative to day zero",
            )
            st.caption(
                "Both allocations use the exact same sampled market rows. The fifth-percentile "
                "line is a downside path summary, not a guaranteed floor."
            )

            st.subheader("5. Did the candidate satisfy every entered limit?")
            st.dataframe(
                allocation_result.constraint_status,
                hide_index=True,
                width="stretch",
                column_config={
                    "constraint": "Constraint",
                    "observed": st.column_config.NumberColumn(
                        "Sandbox value",
                        format="percent",
                    ),
                    "limit": st.column_config.NumberColumn(
                        "Required limit",
                        format="percent",
                    ),
                    "passes": st.column_config.CheckboxColumn("Passes", disabled=True),
                },
            )

            with st.expander(
                "Expected-return estimates and optimizer limitations",
                icon=":material/science:",
            ):
                shrinkage_label = f"{allocation_result.return_shrinkage:.0%}"
                st.markdown(
                    f"Historical mean returns were shrunk **{shrinkage_label}** "
                    "toward the cross-asset average before calculating the displayed return "
                    "estimate. Shrinkage reduces extreme differences; it does not make the "
                    "remaining estimates reliable."
                )
                st.dataframe(
                    allocation_result.expected_return_estimates,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "ticker": "Ticker",
                        "raw_historical_annualized_return": st.column_config.NumberColumn(
                            "Raw historical annualized mean",
                            format="percent",
                        ),
                        "shrunk_annualized_return": st.column_config.NumberColumn(
                            "Shrunk annualized mean",
                            format="percent",
                        ),
                    },
                )
                st.markdown(
                    """
                    - The optimizer is in-sample: it uses the same historical window to estimate
                      risk, returns, and dependence.
                    - Expected returns are much harder to estimate than historical averages make
                      them appear. The risk-adjusted objective is therefore the most fragile.
                    - Tail-loss and drawdown constraints are empirical quantiles from a finite
                      scenario set and can change with the seed, lookback, and bootstrap model.
                    - No taxes, spreads, liquidity, market impact, minimum trade sizes, or
                      account restrictions are included.
                    - The result has not been walk-forward tested and should not be implemented
                      as a trade recommendation.
                    """
                )
            st.error(
                "Do not interpret the sandbox weights as an optimal real-world portfolio. "
                "Use them to understand trade-offs and constraint sensitivity, then validate "
                "on later untouched data and independent stress scenarios.",
                icon=":material/gpp_maybe:",
            )

    with portfolio_audit:
        factor_payload = st.session_state.get("riskforge_factor")
        factor_attribution = None
        factor_stress = None
        if (
            isinstance(factor_payload, dict)
            and factor_payload.get("portfolio_tickers") == portfolio_result.tickers
            and factor_payload.get("portfolio_weights") == tuple(portfolio_result.weights.tolist())
        ):
            candidate_attribution = factor_payload.get("attribution")
            candidate_stress = factor_payload.get("stress")
            if isinstance(candidate_attribution, FactorAttributionResult):
                factor_attribution = candidate_attribution
            if isinstance(candidate_stress, FactorStressResult):
                factor_stress = candidate_stress

        allocation_payload = st.session_state.get("riskforge_allocation")
        allocation_for_audit = None
        if (
            isinstance(allocation_payload, dict)
            and allocation_payload.get("portfolio_tickers") == portfolio_result.tickers
            and allocation_payload.get("portfolio_weights")
            == tuple(portfolio_result.weights.tolist())
            and isinstance(allocation_payload.get("result"), AllocationResult)
        ):
            allocation_for_audit = allocation_payload["result"]

        portfolio_snapshot = create_portfolio_audit(
            portfolio_result,
            quality=portfolio_quality,
            factor_attribution=factor_attribution,
            factor_stress=factor_stress,
            allocation=allocation_for_audit,
        )
        portfolio_file_stem = "riskforge_" + "_".join(
            ticker.replace("^", "index_").replace("=", "_") for ticker in portfolio_result.tickers
        )
        render_audit_panel(
            portfolio_snapshot,
            portfolio_quality,
            file_stem=portfolio_file_stem,
        )
        st.info(
            "A matching factor analysis or allocation sandbox is included only when it was "
            "run for these exact starting holdings and weights.",
            icon=":material/info:",
        )

    with portfolio_limits:
        st.subheader("What the portfolio model assumes")
        shared_return_count = len(portfolio_result.historical_log_returns)
        st.markdown(
            f"""
            - The starting weights total 100% and are **not rebalanced** during each
              {horizon_label(portfolio_result.horizon)} path.
            - Complete return rows are sampled together, preserving same-day relationships
              among the holdings in the selected history.
            - The **{portfolio_result.method.value}** model is used. Moving blocks preserve
              short historical sequences; IID sampling does not.
            - Every ticker must have a valid adjusted price on a shared date. Days missing for
              any holding are removed before the most recent {shared_return_count:,} shared
              returns are selected.
            - Dollar examples assume a starting value of
              **${portfolio_result.initial_value:,.0f}**.
            """
        )
        st.subheader("Important things it does not model")
        st.markdown(
            """
            - Currency conversion or hedging. Returns from different listing currencies are
              treated as quoted, so a mixed-currency portfolio is incomplete without FX risk.
            - Taxes, fees, bid-ask spreads, market impact, liquidity limits, borrowing, short
              positions, options, or leverage.
            - Company fundamentals, valuation, dividends beyond the vendor's adjusted-price
              series, future cash flows, or an estimate of expected return.
            - A correlation breakdown or rare shock that is absent from the chosen history.
            - Automatic rebalancing, deposits, withdrawals, or a rule for what to buy or sell.
            """
        )
        st.error(
            "This analysis does not maximize gains and does not recommend an allocation. A "
            "portfolio should also be tested with explicit historical and hypothetical stress "
            "scenarios before money decisions are made.",
            icon=":material/gpp_maybe:",
        )

    st.subheader("Export this portfolio analysis", icon=":material/download:")
    st.caption(
        "Save a readable report containing the cart allocation, model settings, data checks, "
        "risk results, and limitations."
    )
    st.download_button(
        "Export portfolio analysis report",
        data=audit_html(portfolio_snapshot),
        file_name=f"{portfolio_file_stem}_analysis_report.html",
        mime="text/html",
        on_click="ignore",
        type="primary",
        icon=":material/download:",
        width="stretch",
        key="portfolio_bottom_report_export",
    )
    st.stop()


with st.sidebar:
    st.header("Build your scenario", icon=":material/tune:")
    st.caption("The suggested settings are a good first run.")
    with st.form("simulation_controls"):
        ticker_input = st.text_input(
            "1. Enter any ticker",
            value="SPY",
            max_chars=30,
            placeholder="For example: AAPL",
            icon=":material/search:",
            help=(
                "Enter any symbol supported by Yahoo Finance. A ticker is the short market "
                "name for an investment, such as AAPL for Apple."
            ),
        )
        st.caption(
            "Examples: AAPL (U.S.), SHOP.TO (Canada), VOD.L (U.K.), BTC-USD (crypto), "
            "^GSPC (index), EURUSD=X (FX), GC=F (gold futures)."
        )
        horizon_choice = st.selectbox(
            "2. How long might you hold it?",
            options=list(HORIZONS),
            index=1,
            help="This is how far each simulated future runs.",
        )
        investment_amount = st.number_input(
            "3. Example amount invested ($)",
            min_value=100,
            max_value=100_000_000,
            value=10_000,
            step=500,
            help=(
                "This only translates percentages into dollars. It does not change the "
                "investment's percentage risk."
            ),
        )
        loss_threshold_percent = st.slider(
            "4. What loss would worry you? (%)",
            min_value=0,
            max_value=50,
            value=10,
            step=1,
            help="RiskForge will estimate how often this loss was reached or exceeded.",
        )

        with st.expander("Advanced settings", icon=":material/settings:"):
            model_choice = st.segmented_control(
                "How should past market days be sampled?",
                options=list(MODEL_LABELS),
                default=BootstrapMethod.MOVING_BLOCK.value,
                required=True,
                format_func=lambda option: MODEL_LABELS[str(option)],
                help=(
                    "Consecutive-day blocks retain short runs of calm or turbulent days. "
                    "Independent days are the original simpler baseline."
                ),
                width="stretch",
                wrap=True,
            )
            block_size_input = st.select_slider(
                "Days kept together in each block",
                options=[2, 3, 5, 10, 20],
                value=5,
                help=(
                    "Five trading days is a transparent starting point. This setting is "
                    "ignored when independent days are selected."
                ),
            )
            paths_input = st.select_slider(
                "Number of possible futures",
                options=[1_000, 5_000, 10_000, 25_000, 50_000],
                value=10_000,
                help="More futures make percentages steadier but take more computing time.",
            )
            history_choice = st.selectbox(
                "How much past history should the model learn from?",
                options=list(HISTORY_WINDOWS),
                index=2,
                help=(
                    "A longer window includes more market conditions. A shorter window "
                    "reflects recent behavior more strongly."
                ),
            )
            seed_input = st.number_input(
                "Reproducibility number",
                min_value=0,
                value=42,
                step=1,
                help="Keep this unchanged to reproduce the exact same random sample.",
            )

        submitted = st.form_submit_button(
            "Show me the risk",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    st.caption(
        "Educational research tool only. The results depend on the model and are not "
        "investment advice."
    )

if submitted:
    clean_ticker = str(ticker_input).strip().upper()
    if not clean_ticker:
        st.error("Choose or enter an investment symbol before running the simulation.")
    else:
        try:
            with st.skeleton(height=160):
                simulation_result = run_simulation(
                    clean_ticker,
                    HORIZONS[str(horizon_choice)],
                    int(paths_input),
                    HISTORY_WINDOWS[str(history_choice)],
                    int(seed_input),
                    loss_threshold_percent / 100,
                    str(model_choice),
                    int(block_size_input),
                ).copy()
                simulation_result["investment_amount"] = float(investment_amount)
                st.session_state["riskforge_result"] = simulation_result
                st.session_state["riskforge_backtest"] = None
                st.session_state["riskforge_robustness"] = None
                st.session_state["riskforge_position"] = None
                st.session_state["riskforge_volatility"] = None
                st.session_state["riskforge_ml"] = None
            st.toast("Simulation complete", icon=":material/check_circle:")
        except (MarketDataError, ValueError, TypeError) as exc:
            st.session_state["riskforge_result"] = None
            st.error(str(exc), icon=":material/error:")
        except Exception as exc:  # keep a hosted app useful when a data vendor changes
            st.session_state["riskforge_result"] = None
            st.error(
                f"The simulation could not be completed: {exc}",
                icon=":material/error:",
            )

result = st.session_state.get("riskforge_result")
if result is None:
    st.header("Your first simulation, in three steps", icon=":material/map:")
    step_columns = st.columns(3, border=True)
    with step_columns[0]:
        st.subheader("1. Pick an investment")
        st.markdown(
            "Start with **SPY**, a fund that follows much of the U.S. stock market, or type "
            "another ticker."
        )
    with step_columns[1]:
        st.subheader("2. Describe your concern")
        st.markdown(
            "Choose how long you might hold it, an example dollar amount, and the loss that "
            "would worry you."
        )
    with step_columns[2]:
        st.subheader("3. Read the range")
        st.markdown(
            "RiskForge creates many possible futures and shows how often losses appeared. "
            "No single path is a prediction."
        )

    st.info(
        "For a useful first example, leave the suggested settings unchanged and select "
        "**Show me the risk** in the sidebar.",
        icon=":material/lightbulb:",
    )
    with st.expander("What does Monte Carlo simulation mean?", icon=":material/help:"):
        st.markdown(
            """
            A Monte Carlo simulation is a structured **what-if exercise**. RiskForge takes
            actual daily gains and losses from the investment's history, randomly rearranges
            and reuses either individual days or short consecutive blocks, and builds
            thousands of possible paths.

            It helps answer questions like *"How wide could the range of outcomes be?"* and
            *"How often did a large loss appear under these assumptions?"* It does **not**
            know tomorrow's news and does not tell you what will happen next.
            """
        )
    st.stop()

summary = result["summary"]
prices = result["prices"]
price_paths = result["price_paths"]
assert isinstance(summary, RiskSummary)
assert isinstance(prices, pd.Series)
assert isinstance(price_paths, np.ndarray)
quality = result.get("data_quality")
if not isinstance(quality, DataQualityReport):
    quality = assess_price_quality(
        prices,
        minimum_returns=max(2, min(len(prices) - 1, int(result["lookback"]))),
    )
ticker = str(result["ticker"])
threshold = float(result["loss_threshold"])
investment = float(result.get("investment_amount", 10_000))
horizon_days = int(result["horizon"])
bootstrap_method = BootstrapMethod(str(result.get("method", BootstrapMethod.IID.value)))
block_size = int(result.get("block_size", 1))
configured_model_label = bootstrap_model_label(
    bootstrap_method,
    block_size,
    horizon_days,
)
loss_count = out_of_100(summary.probability_of_loss)
threshold_count = out_of_100(summary.probability_exceeding_threshold)
outcome_split = summarize_outcome_split(price_paths)
historical_stress = analyze_historical_stress(
    prices,
    horizon=horizon_days,
    loss_threshold=threshold,
)
historical_var_exceedance = historical_exceedance_rate(
    historical_stress,
    summary.var_95,
)

st.header(f"What the {ticker} simulation found", icon=":material/insights:")
st.caption(
    f"{int(result['paths']):,} possible futures over {horizon_label(horizon_days)}, using "
    f"{int(result['lookback']):,} past daily returns. Latest adjusted price: "
    f"${float(prices.iloc[-1]):,.2f}. Model: {configured_model_label}."
)

with st.container(border=True):
    st.subheader("Start here: the result in everyday language")
    st.markdown(
        f"""
        - **{loss_count} out of 100** simulated futures ended below the starting price.
        - The cutoff for the **worst 5 out of 100** was {describe_loss(summary.var_95, investment)}.
          About 5 out of 100 simulated outcomes were worse than this cutoff.
        - Those worst 5 out of 100 outcomes averaged
          {describe_loss(summary.expected_shortfall_95, investment)}.
        - **{threshold_count} out of 100** simulated futures lost at least **{threshold:.0%}**,
          the loss level you asked RiskForge to watch.

        Dollar examples assume **${investment:,.0f} invested** and ignore fees, taxes,
        and trading costs.
        """
    )
    st.warning(
        "This is a stress test, not a forecast. It asks what could happen if future daily "
        "moves resemble the historical sample under the selected model; real markets can "
        "behave differently.",
        icon=":material/warning:",
    )
    if bootstrap_method is BootstrapMethod.MOVING_BLOCK:
        st.info(
            f"**Model used:** {configured_model_label}. RiskForge keeps short historical "
            "runs together so a turbulent day can remain beside nearby turbulent days.",
            icon=":material/view_week:",
        )
    else:
        st.info(
            "**Model used:** Independent-day bootstrap. This is the simple baseline and "
            "does not preserve the order of nearby market moves.",
            icon=":material/scatter_plot:",
        )

with st.container(horizontal=True):
    st.metric(
        "Chance of any loss",
        f"{loss_count} in 100",
        help="The share of simulated futures that finished below the starting price.",
        border=True,
    )
    st.metric(
        "Worst 5-in-100 cutoff",
        short_loss_label(summary.var_95),
        help="About 5% of simulated outcomes were worse than this. This is also called 95% VaR.",
        border=True,
    )
    st.metric(
        "Average of worst 5 in 100",
        short_loss_label(summary.expected_shortfall_95),
        help=(
            "The average result among the worst 5% of simulations. Also called Expected Shortfall."
        ),
        border=True,
    )
    st.metric(
        f"Chance of losing {threshold:.0%}+",
        f"{threshold_count} in 100",
        help="How often the simulation reached or exceeded your chosen loss threshold.",
        border=True,
    )

render_outcome_split(outcome_split, investment, subject=ticker)

(
    tab_paths,
    tab_distribution,
    tab_history,
    tab_stress,
    tab_regimes,
    tab_robustness,
    tab_volatility,
    tab_ml,
    tab_position,
    tab_backtest,
    tab_audit,
    tab_details,
) = st.tabs(
    [
        ":material/route: Possible futures",
        ":material/bar_chart: Range of outcomes",
        ":material/history: Past data used",
        ":material/thunderstorm: Historical stress",
        ":material/cyclone: Market regimes",
        ":material/compare_arrows: Compare models",
        ":material/show_chart: Adaptive volatility",
        ":material/model_training: ML lab",
        ":material/pie_chart: Size a position",
        ":material/fact_check: Did the model work?",
        ":material/verified_user: Audit and export",
        ":material/menu_book: Definitions and limits",
    ]
)
with tab_paths:
    st.markdown(
        "**How to read this:** each thin line is one possible price journey. The solid line "
        "is the middle path, not a prediction. Wider shaded areas mean more uncertainty."
    )
    fan_chart = build_fan_chart(price_paths, ticker, sample_paths=20)
    fan_chart.update_layout(
        title=f"Many possible {ticker} price paths over {horizon_label(horizon_days)}",
        template="plotly_dark",
        height=620,
    )
    st.plotly_chart(fan_chart, width="stretch", config={"displaylogo": False})
    st.caption(
        "The darker band contains the middle half of paths. The lighter band contains the "
        "middle 90%, so roughly 1 in 10 paths lies outside it."
    )

with tab_distribution:
    st.markdown(
        "**How to read this:** every bar counts simulated futures that ended with a similar "
        "gain or loss. Outcomes on the left are losses; outcomes on the right are gains."
    )
    st.plotly_chart(
        build_distribution_chart(price_paths, ticker),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "A tall bar means that outcome appeared frequently in the simulation. It does not "
        "mean the market is guaranteed to land there."
    )

with tab_history:
    st.markdown(
        "**Why this matters:** these are the past prices the model learned from. Changing the "
        "history window can change the result because different years contain different risks."
    )
    st.plotly_chart(
        build_history_chart(prices, ticker),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        f"The sample runs through {prices.index[-1].date()} and contains "
        f"{len(prices) - 1:,} daily returns. Adjusted prices account for splits and "
        "distributions as supplied by Yahoo Finance."
    )

with tab_stress:
    if bootstrap_method is BootstrapMethod.MOVING_BLOCK:
        stress_model_explanation = (
            "the Monte Carlo simulation stitches together short historical blocks"
        )
    else:
        stress_model_explanation = (
            "the Monte Carlo simulation rearranges individual historical days"
        )
    st.markdown(
        f"**What this checks:** {stress_model_explanation}. This view keeps the entire market "
        "sequence in its original order and asks what actually happened over rolling "
        f"{horizon_label(horizon_days)} periods."
    )
    st.caption(
        "Rolling periods overlap, so these frequencies describe this sample; they are not "
        "independent observations or promises about future probabilities."
    )

    if historical_stress.observations < 40:
        st.warning(
            "This history provides very few periods at the selected holding length. Use a "
            "longer history window or shorter holding period for a more useful comparison.",
            icon=":material/hourglass:",
        )

    with st.container(border=True):
        st.subheader("Monte Carlo compared with actual history")
        actual_exceedance_count = out_of_100(historical_var_exceedance)
        st.markdown(
            f"The simulation's worst-5-in-100 cutoff was "
            f"**{short_loss_label(summary.var_95)}**. In the real ordered history, about "
            f"**{actual_exceedance_count} in 100** overlapping {horizon_days}-day periods "
            "were worse than that cutoff."
        )
        if historical_var_exceedance > 0.05:
            st.warning(
                "Actual historical sequences crossed the simulated cutoff more often than "
                "5 in 100. The selected simulation may still be smoothing over runs of bad "
                "days or changing market regimes.",
                icon=":material/warning:",
            )
        else:
            st.info(
                "This sample did not cross the simulated 95% cutoff more than 5 in 100 "
                "times. That is a useful check, not proof that the cutoff is future-safe.",
                icon=":material/info:",
            )

    with st.container(horizontal=True):
        st.metric(
            "Worst actual outcome",
            short_loss_label(historical_stress.worst_loss),
            help=(
                f"From {historical_stress.worst_start_date.date()} through "
                f"{historical_stress.worst_end_date.date()}."
            ),
            border=True,
        )
        st.metric(
            "Average of actual worst 5%",
            short_loss_label(historical_stress.historical_expected_shortfall_95),
            help="Average loss among the worst 5% of rolling historical periods.",
            border=True,
        )
        st.metric(
            "Past periods beyond simulated VaR",
            f"{out_of_100(historical_var_exceedance)} in 100",
            help="Overlapping historical periods worse than the simulated 95% VaR cutoff.",
            border=True,
        )
        st.metric(
            f"Past periods losing {threshold:.0%}+",
            f"{out_of_100(historical_stress.probability_exceeding_threshold)} in 100",
            help="Overlapping historical periods that crossed your watched loss level.",
            border=True,
        )

    st.plotly_chart(
        build_historical_stress_chart(
            historical_stress,
            summary.var_95,
            summary.var_99,
            ticker,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "A point above a cutoff line is an actual historical period whose loss was worse "
        "than that simulated risk estimate. Diamonds mark distinct severe episodes."
    )

    st.subheader("Worst distinct historical episodes")
    episode_table = historical_stress.worst_episodes.copy()
    episode_table["start_date"] = pd.to_datetime(episode_table["start_date"]).dt.date
    episode_table["end_date"] = pd.to_datetime(episode_table["end_date"]).dt.date
    episode_table["hypothetical_change"] = episode_table["return"] * investment
    st.dataframe(
        episode_table[["start_date", "end_date", "return", "hypothetical_change"]],
        hide_index=True,
        width="stretch",
        column_config={
            "start_date": st.column_config.DateColumn("Started", format="MMM D, YYYY"),
            "end_date": st.column_config.DateColumn("Ended", format="MMM D, YYYY"),
            "return": st.column_config.NumberColumn("Gain or loss", format="percent"),
            "hypothetical_change": st.column_config.NumberColumn(
                f"Change on ${investment:,.0f}",
                format="dollar",
            ),
        },
    )
    st.caption(
        "Overlapping windows are removed from this table so one extended selloff is not "
        "listed repeatedly. Dollar changes are hypothetical and ignore costs and taxes."
    )

with tab_regimes:
    st.markdown(
        "**What this checks:** markets can behave differently during calm advances, calm "
        "declines, turbulent advances, and turbulent declines. RiskForge labels those states "
        "using a rolling three-month trend and volatility, then examines what followed."
    )
    st.warning(
        "A market-state label is context, not a buy or sell signal. Regimes can change "
        "quickly, and similar-looking periods can end very differently.",
        icon=":material/warning:",
    )
    regime_outlook_choice = st.selectbox(
        "How far ahead should historical outcomes be compared?",
        options=list(REGIME_OUTLOOKS),
        index=1,
        key="regime_outlook",
        help=(
            "This does not forecast the future. It measures returns following similar "
            "historical labels."
        ),
    )
    regime_analysis = analyze_market_regimes(
        prices,
        window=63,
        forward_horizon=REGIME_OUTLOOKS[str(regime_outlook_choice)],
    )

    with st.container(border=True):
        st.subheader("Current historical label")
        if regime_analysis.current_regime is MarketRegime.CALM_ADVANCE:
            regime_explanation = (
                "The recent three-month trend is positive and volatility is below its "
                "typical level in this sample."
            )
        elif regime_analysis.current_regime is MarketRegime.CALM_DECLINE:
            regime_explanation = (
                "The recent three-month trend is negative while volatility remains below "
                "its typical level in this sample."
            )
        elif regime_analysis.current_regime is MarketRegime.TURBULENT_ADVANCE:
            regime_explanation = (
                "The recent three-month trend is positive, but price movement is more "
                "volatile than usual for this sample."
            )
        else:
            regime_explanation = (
                "The recent three-month trend is negative and price movement is more "
                "volatile than usual for this sample."
            )
        st.markdown(regime_explanation)
        with st.container(horizontal=True):
            st.metric(
                "Current regime",
                regime_analysis.current_regime.value,
                border=True,
            )
            st.metric(
                "Recent three-month trend",
                f"{regime_analysis.current_rolling_return:.1%}",
                border=True,
            )
            st.metric(
                "Recent annualized volatility",
                f"{regime_analysis.current_annualized_volatility:.1%}",
                help="Annualized using the common 252-trading-day convention.",
                border=True,
            )
            st.metric(
                "Calm/turbulent boundary",
                f"{regime_analysis.volatility_threshold:.1%}",
                help="The median rolling volatility observed up through the latest date.",
                border=True,
            )

    matching_regime = regime_analysis.summary[
        regime_analysis.summary["regime"] == regime_analysis.current_regime.value
    ]
    with st.container(border=True):
        st.subheader("What followed similar historical labels")
        if matching_regime.empty:
            st.info(
                "There are no completed forward periods for this state in the selected "
                "sample. A longer history is needed.",
                icon=":material/hourglass:",
            )
        else:
            current_history = matching_regime.iloc[0]
            current_observations = int(current_history["observations"])
            st.markdown(
                f"RiskForge found **{current_observations} overlapping historical "
                f"observations** labeled **{regime_analysis.current_regime.value}** with a "
                f"completed {regime_outlook_choice} outcome."
            )
            with st.container(horizontal=True):
                st.metric(
                    "Middle subsequent outcome",
                    f"{float(current_history['median_forward_return']):.1%}",
                    border=True,
                )
                st.metric(
                    "Subsequent periods with a loss",
                    f"{out_of_100(float(current_history['probability_of_loss']))} in 100",
                    border=True,
                )
                st.metric(
                    "Average of worst subsequent 5%",
                    short_loss_label(float(current_history["expected_shortfall_95"])),
                    border=True,
                )
            if current_observations < 20:
                st.warning(
                    "Fewer than 20 similar observations were available, so these figures "
                    "are especially fragile.",
                    icon=":material/hourglass:",
                )

    st.plotly_chart(
        build_regime_chart(regime_analysis, ticker),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "At each date, turbulent means rolling volatility was above the historical median "
        "available by that date—not that the period was necessarily a crash."
    )

    st.subheader(f"Outcomes {regime_outlook_choice} after each historical label")
    regime_table = regime_analysis.summary.copy()
    st.dataframe(
        regime_table,
        hide_index=True,
        width="stretch",
        column_config={
            "regime": "Historical label",
            "observations": st.column_config.NumberColumn("Observations", format="%d"),
            "sample_share": st.column_config.NumberColumn("Share of sample", format="percent"),
            "median_forward_return": st.column_config.NumberColumn(
                "Middle subsequent outcome",
                format="percent",
            ),
            "probability_of_loss": st.column_config.NumberColumn(
                "Subsequent loss frequency",
                format="percent",
            ),
            "var_95": st.column_config.NumberColumn("95% loss cutoff", format="percent"),
            "expected_shortfall_95": st.column_config.NumberColumn(
                "Average worst 5% loss",
                format="percent",
            ),
            "worst_loss": st.column_config.NumberColumn(
                "Worst observed loss",
                format="percent",
            ),
        },
    )
    st.caption(
        "Forward periods overlap. Labels use information available through each historical "
        "date, but this remains descriptive analysis—not an out-of-sample prediction."
    )

with tab_robustness:
    st.markdown(
        "**What this checks:** the result should not depend on one convenient set of "
        "assumptions. RiskForge reruns the analysis with independent days, several "
        "consecutive-day block sizes, and several lengths of price history."
    )
    st.warning(
        "A higher or lower number does not automatically make one model correct. Large "
        "differences mean **model choice itself is an important source of uncertainty**.",
        icon=":material/balance:",
    )

    robustness_horizon_labels = list(REGIME_OUTLOOKS)
    current_horizon_label = horizon_label(horizon_days)
    robustness_default_index = (
        robustness_horizon_labels.index(current_horizon_label)
        if current_horizon_label in robustness_horizon_labels
        else 1
    )
    with st.form("robustness_controls"):
        robustness_columns = st.columns(3)
        with robustness_columns[0]:
            robustness_horizon_choice = st.selectbox(
                "Comparison horizon",
                options=robustness_horizon_labels,
                index=robustness_default_index,
                help=(
                    "Shorter horizons leave more non-overlapping periods for the historical "
                    "model check."
                ),
            )
        with robustness_columns[1]:
            robustness_paths = st.select_slider(
                "Simulations per comparison",
                options=[500, 1_000, 2_000, 5_000],
                value=1_000,
                help="More paths reduce random simulation noise but take longer.",
            )
        with robustness_columns[2]:
            robustness_seed = st.number_input(
                "Comparison reproducibility number",
                min_value=0,
                value=42,
                step=1,
            )
        robustness_submitted = st.form_submit_button(
            "Compare model assumptions",
            type="primary",
            icon=":material/compare_arrows:",
        )

    if robustness_submitted:
        try:
            with st.skeleton(height=180):
                robustness_result = run_robustness_analysis(
                    prices=prices,
                    horizon=REGIME_OUTLOOKS[str(robustness_horizon_choice)],
                    paths=int(robustness_paths),
                    seed=int(robustness_seed),
                    loss_threshold=threshold,
                    selected_method=bootstrap_method.value,
                    selected_block_size=block_size,
                )
                st.session_state["riskforge_robustness"] = {
                    "ticker": ticker,
                    "result": robustness_result,
                }
            st.toast("Model comparison complete", icon=":material/check_circle:")
        except (ValueError, TypeError) as exc:
            st.session_state["riskforge_robustness"] = None
            st.error(str(exc), icon=":material/error:")
        except Exception as exc:
            st.session_state["riskforge_robustness"] = None
            st.error(f"The model comparison failed: {exc}", icon=":material/error:")

    robustness_payload = st.session_state.get("riskforge_robustness")
    if not robustness_payload or robustness_payload["ticker"] != ticker:
        st.info(
            "Select **Compare model assumptions** to see how sensitive the answer is to "
            "the sampling model and historical window.",
            icon=":material/lightbulb:",
        )
    else:
        compared_result = robustness_payload["result"]
        assert isinstance(compared_result, RobustnessResult)
        model_comparison = compared_result.model_comparison.copy()
        lookback_comparison = compared_result.lookback_comparison.copy()
        backtest_comparison = compared_result.backtest_comparison.copy()

        largest_var_row = model_comparison.loc[model_comparison["var_95"].idxmax()]
        model_var_spread = float(
            model_comparison["var_95"].max() - model_comparison["var_95"].min()
        )
        lookback_var_spread = float(
            lookback_comparison["var_95"].max() - lookback_comparison["var_95"].min()
        )
        with st.container(horizontal=True):
            st.metric(
                "Models compared",
                f"{len(model_comparison)}",
                help="One independent-day baseline plus distinct moving-block choices.",
                border=True,
            )
            st.metric(
                "Largest 95% loss cutoff",
                short_loss_label(float(largest_var_row["var_95"])),
                help=f"Produced by {largest_var_row['model']} in this run.",
                border=True,
            )
            st.metric(
                "Model VaR spread",
                f"{model_var_spread:.1%}",
                help="Highest minus lowest 95% VaR across sampling assumptions.",
                border=True,
            )
            st.metric(
                "Lookback VaR spread",
                f"{lookback_var_spread:.1%}",
                help="Highest minus lowest 95% VaR across available history lengths.",
                border=True,
            )

        st.subheader("1. Does the sampling assumption change the answer?")
        st.markdown(
            "Independent days discard the order of past returns. Moving blocks retain short "
            "runs, with longer blocks preserving longer runs. Material disagreement is a "
            "reason to plan around a range instead of trusting one estimate."
        )
        model_chart = model_comparison.set_index("model")[["var_95", "expected_shortfall_95"]]
        model_chart.columns = ["95% VaR", "95% Expected Shortfall"]
        st.bar_chart(
            model_chart,
            y=["95% VaR", "95% Expected Shortfall"],
            y_label="Loss fraction",
        )
        st.dataframe(
            model_comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "model": "Sampling model",
                "method": "Method",
                "block_size": st.column_config.NumberColumn("Block size", format="%d"),
                "var_95": st.column_config.NumberColumn("95% VaR", format="percent"),
                "var_99": st.column_config.NumberColumn("99% VaR", format="percent"),
                "expected_shortfall_95": st.column_config.NumberColumn(
                    "Average worst 5% loss",
                    format="percent",
                ),
                "probability_of_loss": st.column_config.NumberColumn(
                    "Chance of loss",
                    format="percent",
                ),
                "probability_exceeding_threshold": st.column_config.NumberColumn(
                    f"Chance of losing {threshold:.0%}+",
                    format="percent",
                ),
                "maximum_drawdown_95": st.column_config.NumberColumn(
                    "95% drawdown cutoff",
                    format="percent",
                ),
            },
        )

        st.subheader("2. Did each model's 95% cutoff match past outcomes?")
        st.markdown(
            "Each row uses the same walk-forward test: only earlier data are available at "
            "each checkpoint. A p-value below 0.05 is a warning flag, while a larger value "
            "does **not** prove the model is safe."
        )
        coverage_labels = {
            "limited_evidence": "Limited evidence",
            "risk_underestimated": "Risk underestimated",
            "risk_overestimated": "Risk overestimated",
            "no_failure_detected": "No coverage failure detected",
        }
        independence_labels = {
            "limited_evidence": "Limited evidence",
            "clustered": "Violations clustered",
            "dependence_detected": "Dependence detected",
            "no_clustering_detected": "No dependence detected",
        }
        backtest_comparison["coverage_status"] = backtest_comparison["coverage_status"].replace(
            coverage_labels
        )
        backtest_comparison["independence_status"] = backtest_comparison[
            "independence_status"
        ].replace(independence_labels)
        st.dataframe(
            backtest_comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "model": "Sampling model",
                "observations": st.column_config.NumberColumn("Periods", format="%d"),
                "violations": st.column_config.NumberColumn("Violations", format="%d"),
                "expected_violations": st.column_config.NumberColumn(
                    "Expected violations",
                    format="%.1f",
                ),
                "violation_rate": st.column_config.NumberColumn(
                    "Violation rate",
                    format="percent",
                ),
                "kupiec_p_value": st.column_config.NumberColumn(
                    "Coverage p-value",
                    format="%.3f",
                ),
                "independence_p_value": st.column_config.NumberColumn(
                    "Independence p-value",
                    format="%.3f",
                ),
                "conditional_coverage_p_value": st.column_config.NumberColumn(
                    "Combined p-value",
                    format="%.3f",
                ),
                "coverage_status": "Coverage reading",
                "independence_status": "Clustering reading",
                "longest_violation_run": st.column_config.NumberColumn(
                    "Longest violation run",
                    format="%d",
                ),
            },
        )
        st.caption(
            f"The walk-forward check trained on up to {compared_result.training_window:,} "
            f"daily returns and had {compared_result.evaluation_returns:,} returns available "
            "for non-overlapping evaluation periods. Longer horizons produce fewer tests."
        )

        st.subheader("3. Does the amount of history change the answer?")
        selected_comparison_model = bootstrap_model_label(
            bootstrap_method,
            block_size,
            compared_result.horizon,
        )
        st.markdown(
            f"This holds the selected **{selected_comparison_model}** fixed and changes only "
            "how much of the most recent price history it sees."
        )
        lookback_chart = lookback_comparison.set_index("lookback")[
            ["var_95", "expected_shortfall_95"]
        ]
        lookback_chart.columns = ["95% VaR", "95% Expected Shortfall"]
        st.line_chart(
            lookback_chart,
            y=["95% VaR", "95% Expected Shortfall"],
            x_label="Historical daily returns used",
            y_label="Loss fraction",
        )
        st.dataframe(
            lookback_comparison[
                [
                    "model",
                    "lookback",
                    "var_95",
                    "var_99",
                    "expected_shortfall_95",
                    "probability_of_loss",
                    "maximum_drawdown_95",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "model": "History window",
                "lookback": st.column_config.NumberColumn("Daily returns", format="%d"),
                "var_95": st.column_config.NumberColumn("95% VaR", format="percent"),
                "var_99": st.column_config.NumberColumn("99% VaR", format="percent"),
                "expected_shortfall_95": st.column_config.NumberColumn(
                    "Average worst 5% loss",
                    format="percent",
                ),
                "probability_of_loss": st.column_config.NumberColumn(
                    "Chance of loss",
                    format="percent",
                ),
                "maximum_drawdown_95": st.column_config.NumberColumn(
                    "95% drawdown cutoff",
                    format="percent",
                ),
            },
        )
        if len(lookback_comparison) == 1:
            st.warning(
                "Only one useful history window is available. Rerun the main simulation "
                "with a longer history to make this sensitivity check informative.",
                icon=":material/hourglass:",
            )
        st.caption(
            f"All comparisons used {compared_result.paths:,} paths and reproducibility "
            f"number {compared_result.seed}. A fixed seed makes the experiment repeatable; "
            "it does not remove sampling uncertainty."
        )

with tab_volatility:
    st.markdown(
        "**What this checks:** the ordinary bootstrap treats every historical return as if "
        "it came from today's volatility. The adaptive model first separates return shocks "
        "from their historical volatility, then rescales those shocks to current conditions."
    )
    st.warning(
        "Volatility-aware does not automatically mean more accurate. The walk-forward test "
        "below is the deciding evidence for this historical sample, and even a better result "
        "does not guarantee future performance.",
        icon=":material/science:",
    )

    volatility_horizon_labels = list(REGIME_OUTLOOKS)
    volatility_current_horizon = horizon_label(horizon_days)
    volatility_default_index = (
        volatility_horizon_labels.index(volatility_current_horizon)
        if volatility_current_horizon in volatility_horizon_labels
        else 1
    )
    available_return_count = len(prices) - 1
    volatility_training_window = min(756, max(60, available_return_count // 2))
    with st.form("volatility_controls"):
        volatility_columns = st.columns(3)
        with volatility_columns[0]:
            volatility_horizon_choice = st.selectbox(
                "Adaptive forecast horizon",
                options=volatility_horizon_labels,
                index=volatility_default_index,
            )
            volatility_paths = st.select_slider(
                "Simulations at each checkpoint",
                options=[500, 1_000, 2_000, 5_000],
                value=1_000,
                help="More paths reduce simulation noise but make the walk-forward test slower.",
            )
        with volatility_columns[1]:
            volatility_decay_choice = st.selectbox(
                "How quickly should volatility react?",
                options=list(EWMA_DECAYS),
                index=1,
                help=(
                    "Lower decay reacts faster to recent shocks. Higher decay changes more "
                    "slowly. The 0.94 setting is the classic daily RiskMetrics convention."
                ),
            )
            volatility_seed = st.number_input(
                "Adaptive-model reproducibility number",
                min_value=0,
                value=42,
                step=1,
            )
        with volatility_columns[2]:
            st.metric(
                "Training returns per checkpoint",
                f"{volatility_training_window:,}",
                help=(
                    "Every forecast uses only this many immediately preceding returns. "
                    "Later outcomes are withheld."
                ),
                border=True,
            )
            st.caption(
                "The remaining history is evaluated in non-overlapping forecast periods. "
                "RiskForge tests 95% VaR so the sample has more expected violations than a "
                "99% test."
            )
        volatility_submitted = st.form_submit_button(
            "Run volatility-aware comparison",
            type="primary",
            icon=":material/show_chart:",
        )

    if volatility_submitted:
        try:
            with st.skeleton(height=180):
                volatility_result = run_volatility_analysis(
                    prices=prices,
                    horizon=REGIME_OUTLOOKS[str(volatility_horizon_choice)],
                    paths=int(volatility_paths),
                    training_window=volatility_training_window,
                    seed=int(volatility_seed),
                    decay=EWMA_DECAYS[str(volatility_decay_choice)],
                    loss_threshold=threshold,
                    method=bootstrap_method.value,
                    block_size=block_size,
                )
                st.session_state["riskforge_volatility"] = {
                    "ticker": ticker,
                    "result": volatility_result,
                }
            st.toast("Volatility-aware comparison complete", icon=":material/check_circle:")
        except (ValueError, TypeError) as exc:
            st.session_state["riskforge_volatility"] = None
            st.error(str(exc), icon=":material/error:")
        except Exception as exc:
            st.session_state["riskforge_volatility"] = None
            st.error(f"The volatility-aware comparison failed: {exc}", icon=":material/error:")

    volatility_payload = st.session_state.get("riskforge_volatility")
    if not volatility_payload or volatility_payload["ticker"] != ticker:
        st.info(
            "Select **Run volatility-aware comparison** to compare the adaptive model with "
            "the selected historical-bootstrap baseline.",
            icon=":material/lightbulb:",
        )
    else:
        adaptive_result = volatility_payload["result"]
        assert isinstance(adaptive_result, VolatilityAnalysisResult)
        current_comparison = adaptive_result.current_comparison.copy()
        fixed_row = current_comparison.loc[
            current_comparison["model"] == "Fixed historical bootstrap"
        ].iloc[0]
        adaptive_row = current_comparison.loc[
            current_comparison["model"] == "EWMA volatility-aware"
        ].iloc[0]
        volatility_ratio = (
            adaptive_result.current_annualized_volatility
            / adaptive_result.median_annualized_volatility
        )

        with st.container(horizontal=True):
            st.metric(
                "Current EWMA volatility",
                f"{adaptive_result.current_annualized_volatility:.1%}",
                help="Annualized from the next-day EWMA volatility estimate.",
                border=True,
            )
            st.metric(
                "Historical median volatility",
                f"{adaptive_result.median_annualized_volatility:.1%}",
                border=True,
            )
            st.metric(
                "Current versus median",
                f"{volatility_ratio:.2f}×",
                help="Above 1 means current modeled volatility is above its historical median.",
                border=True,
            )
            st.metric(
                "Adaptive 95% loss cutoff",
                short_loss_label(float(adaptive_row["var_95"])),
                delta=f"{float(adaptive_row['var_95'] - fixed_row['var_95']):.1%} vs fixed",
                delta_color="inverse",
                border=True,
            )

        st.subheader("1. How volatility changed through the sample")
        volatility_chart = adaptive_result.volatility_history.copy()
        volatility_chart["historical_median"] = adaptive_result.median_annualized_volatility
        volatility_chart = volatility_chart.rename(
            columns={
                "annualized_volatility": "EWMA annualized volatility",
                "historical_median": "Full-sample median reference",
            }
        )
        st.line_chart(
            volatility_chart,
            x="date",
            y=["EWMA annualized volatility", "Full-sample median reference"],
            y_label="Annualized volatility",
        )
        half_life = np.log(0.5) / np.log(adaptive_result.decay)
        st.caption(
            f"Decay {adaptive_result.decay:.2f} gives shocks a volatility half-life of about "
            f"{half_life:.1f} trading days. The median line is descriptive and is not used "
            "to generate forecasts."
        )

        st.subheader("2. Today's fixed versus adaptive scenarios")
        st.markdown(
            "Both models reuse empirical historical observations and the same random seed. "
            "The adaptive version resamples standardized shocks and updates volatility along "
            "each path; the fixed version resamples returns directly."
        )
        current_chart = current_comparison.set_index("model")[["var_95", "expected_shortfall_95"]]
        current_chart.columns = ["95% VaR", "95% Expected Shortfall"]
        st.bar_chart(
            current_chart,
            y=["95% VaR", "95% Expected Shortfall"],
            y_label="Loss fraction",
        )
        st.dataframe(
            current_comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "model": "Model",
                "var_95": st.column_config.NumberColumn("95% VaR", format="percent"),
                "var_99": st.column_config.NumberColumn("99% VaR", format="percent"),
                "expected_shortfall_95": st.column_config.NumberColumn(
                    "Average worst 5% loss",
                    format="percent",
                ),
                "expected_shortfall_99": st.column_config.NumberColumn(
                    "Average worst 1% loss",
                    format="percent",
                ),
                "probability_of_loss": st.column_config.NumberColumn(
                    "Chance of loss",
                    format="percent",
                ),
                "probability_exceeding_threshold": st.column_config.NumberColumn(
                    f"Chance of losing {threshold:.0%}+",
                    format="percent",
                ),
                "maximum_drawdown_95": st.column_config.NumberColumn(
                    "95% drawdown cutoff",
                    format="percent",
                ),
            },
        )
        adaptive_fan_chart = build_fan_chart(
            adaptive_result.adaptive_price_paths,
            ticker,
            sample_paths=20,
        )
        adaptive_fan_chart.update_layout(
            title=(
                f"Volatility-aware {ticker} paths over {horizon_label(adaptive_result.horizon)}"
            ),
            template="plotly_dark",
            height=560,
        )
        st.plotly_chart(
            adaptive_fan_chart,
            width="stretch",
            config={"displaylogo": False},
        )

        st.subheader("3. Strict walk-forward comparison")
        st.markdown(
            "At each checkpoint, both models see exactly the same preceding training window. "
            "They forecast the next period, RiskForge reveals that outcome, and then moves "
            "forward by a full horizon so evaluation periods do not overlap."
        )
        baseline_check = adaptive_result.baseline_backtest
        adaptive_check = adaptive_result.adaptive_backtest
        expected_rate = baseline_check.expected_violation_rate
        baseline_gap = abs(baseline_check.observed_violation_rate - expected_rate)
        adaptive_gap = abs(adaptive_check.observed_violation_rate - expected_rate)
        if np.isclose(baseline_gap, adaptive_gap, atol=0.005):
            st.info(
                "The two models had similarly close violation rates in this evaluation sample. "
                "That is not evidence that they will remain equivalent.",
                icon=":material/balance:",
            )
        elif adaptive_gap < baseline_gap:
            st.info(
                "The volatility-aware model's violation rate was closer to the expected 5% "
                "in this sample. This is descriptive—not proof that it is the better future model.",
                icon=":material/trending_up:",
            )
        else:
            st.warning(
                "The fixed bootstrap's violation rate was closer to the expected 5% in this "
                "sample. Added complexity did not improve this particular coverage result.",
                icon=":material/balance:",
            )

        with st.container(horizontal=True):
            st.metric(
                "Periods tested",
                f"{baseline_check.observations}",
                border=True,
            )
            st.metric(
                "Expected violations",
                f"{baseline_check.expected_violations:.1f}",
                border=True,
            )
            st.metric(
                "Fixed-model violations",
                f"{baseline_check.violations} ({baseline_check.observed_violation_rate:.1%})",
                border=True,
            )
            st.metric(
                "Adaptive-model violations",
                f"{adaptive_check.violations} ({adaptive_check.observed_violation_rate:.1%})",
                border=True,
            )

        coverage_labels = {
            "limited_evidence": "Limited evidence",
            "risk_underestimated": "Risk underestimated",
            "risk_overestimated": "Risk overestimated",
            "no_failure_detected": "No coverage failure detected",
        }
        volatility_backtest_table = pd.DataFrame(
            [
                {
                    "model": "Fixed historical bootstrap",
                    "observations": baseline_check.observations,
                    "violations": baseline_check.violations,
                    "expected_violations": baseline_check.expected_violations,
                    "violation_rate": baseline_check.observed_violation_rate,
                    "coverage_gap": baseline_gap,
                    "coverage_p_value": baseline_check.kupiec_p_value,
                    "independence_p_value": baseline_check.independence_p_value,
                    "combined_p_value": baseline_check.conditional_coverage_p_value,
                    "coverage_status": coverage_labels[baseline_check.status],
                },
                {
                    "model": "EWMA volatility-aware",
                    "observations": adaptive_check.observations,
                    "violations": adaptive_check.violations,
                    "expected_violations": adaptive_check.expected_violations,
                    "violation_rate": adaptive_check.observed_violation_rate,
                    "coverage_gap": adaptive_gap,
                    "coverage_p_value": adaptive_check.kupiec_p_value,
                    "independence_p_value": adaptive_check.independence_p_value,
                    "combined_p_value": adaptive_check.conditional_coverage_p_value,
                    "coverage_status": coverage_labels[adaptive_check.status],
                },
            ]
        )
        st.dataframe(
            volatility_backtest_table,
            hide_index=True,
            width="stretch",
            column_config={
                "model": "Model",
                "observations": st.column_config.NumberColumn("Periods", format="%d"),
                "violations": st.column_config.NumberColumn("Violations", format="%d"),
                "expected_violations": st.column_config.NumberColumn(
                    "Expected violations",
                    format="%.1f",
                ),
                "violation_rate": st.column_config.NumberColumn(
                    "Violation rate",
                    format="percent",
                ),
                "coverage_gap": st.column_config.NumberColumn(
                    "Distance from expected rate",
                    format="percent",
                ),
                "coverage_p_value": st.column_config.NumberColumn(
                    "Coverage p-value",
                    format="%.3f",
                ),
                "independence_p_value": st.column_config.NumberColumn(
                    "Independence p-value",
                    format="%.3f",
                ),
                "combined_p_value": st.column_config.NumberColumn(
                    "Combined p-value",
                    format="%.3f",
                ),
                "coverage_status": "Coverage reading",
            },
        )
        st.plotly_chart(
            build_volatility_backtest_chart(adaptive_result),
            width="stretch",
            config={"displaylogo": False},
        )
        if baseline_check.expected_violations < 5:
            st.warning(
                "This evaluation contains fewer than five expected violations, so statistical "
                "power is limited. Use a longer history or shorter horizon before drawing a "
                "strong conclusion.",
                icon=":material/hourglass:",
            )
        st.caption(
            f"Each forecast trained on {adaptive_result.training_window:,} preceding returns. "
            f"There were {adaptive_result.evaluation_returns:,} later returns available for "
            "non-overlapping evaluation. Repeatedly choosing decay settings after viewing the "
            "same results would overfit this backtest."
        )

        with st.expander("How the adaptive model works", icon=":material/schema:"):
            st.markdown(
                """
                1. At each forecast checkpoint, fit an EWMA volatility filter using only its
                   preceding training window.
                2. Divide training returns by their conditional volatility to obtain empirical
                   shocks.
                3. Resample those shocks using the selected IID or moving-block method.
                4. Scale the shocks by today's estimated volatility.
                5. Update volatility after every simulated shock along every path.

                EWMA has one fixed decay parameter and is intentionally simpler than GARCH or a
                machine-learning model. It can adapt to volatility clustering, but it cannot
                predict news, structural breaks, or a shock absent from the historical sample.
                """
            )

with tab_ml:
    st.markdown(
        "**What this tests:** a ridge-regression model learns whether short-, medium-, and "
        "longer-term volatility patterns improved forecasts of the next period's realized "
        "volatility. It must beat the simpler EWMA benchmark on later, untouched periods."
    )
    st.warning(
        "Machine learning is not automatically smarter or safer. If ridge regression loses "
        "to EWMA out of sample, the simpler benchmark is the stronger result for this test.",
        icon=":material/warning:",
    )

    ml_horizons = {"1 week": 5, "1 month": 20}
    with st.form("ml_volatility_controls"):
        ml_columns = st.columns(3)
        with ml_columns[0]:
            ml_horizon_choice = st.selectbox(
                "ML forecast horizon",
                options=list(ml_horizons),
                index=0,
                help=(
                    "Targets never overlap. A one-month horizon therefore needs much more "
                    "history to create enough independent examples."
                ),
            )
        with ml_columns[1]:
            ml_paths = st.select_slider(
                "ML-scaled Monte Carlo paths",
                options=[500, 1_000, 2_000, 5_000],
                value=1_000,
            )
        with ml_columns[2]:
            ml_seed = st.number_input(
                "ML-lab reproducibility number",
                min_value=0,
                value=42,
                step=1,
            )
        st.caption(
            "Model strength is chosen automatically using only the initial training sample. "
            "RiskForge requires at least 30 non-overlapping examples and keeps the final 40% "
            "for expanding walk-forward evaluation."
        )
        ml_submitted = st.form_submit_button(
            "Train and test the ML model",
            type="primary",
            icon=":material/model_training:",
        )

    if ml_submitted:
        try:
            with st.skeleton(height=180):
                ml_result = run_ml_analysis(
                    prices=prices,
                    horizon=ml_horizons[str(ml_horizon_choice)],
                    paths=int(ml_paths),
                    seed=int(ml_seed),
                    loss_threshold=threshold,
                    method=bootstrap_method.value,
                    block_size=block_size,
                )
                st.session_state["riskforge_ml"] = {
                    "ticker": ticker,
                    "result": ml_result,
                }
            st.toast("ML walk-forward evaluation complete", icon=":material/check_circle:")
        except (ValueError, TypeError) as exc:
            st.session_state["riskforge_ml"] = None
            st.error(str(exc), icon=":material/error:")
        except Exception as exc:
            st.session_state["riskforge_ml"] = None
            st.error(f"The ML evaluation failed: {exc}", icon=":material/error:")

    ml_payload = st.session_state.get("riskforge_ml")
    if not ml_payload or ml_payload["ticker"] != ticker:
        st.info(
            "Select **Train and test the ML model** to evaluate it before viewing an "
            "ML-scaled simulation. Use at least three years of history for the one-month model.",
            icon=":material/lightbulb:",
        )
    else:
        trained_result = ml_payload["result"]
        assert isinstance(trained_result, MlVolatilityResult)
        evaluation = trained_result.evaluation
        if evaluation.ewma_mae > 0:
            relative_mae_change = (evaluation.ewma_mae - evaluation.ml_mae) / evaluation.ewma_mae
        else:
            relative_mae_change = 0.0
        current_volatility_difference = (
            trained_result.current_forecast_annualized_volatility
            - trained_result.current_ewma_annualized_volatility
        )

        if evaluation.ml_mae < evaluation.ewma_mae:
            st.info(
                f"Ridge ML had {relative_mae_change:.1%} lower mean absolute error than EWMA "
                "on the untouched walk-forward sample. This is encouraging evidence for this "
                "ticker and period—not a guarantee of future superiority.",
                icon=":material/experiment:",
            )
        else:
            st.warning(
                f"Ridge ML had {abs(relative_mae_change):.1%} higher mean absolute error than "
                "EWMA on the untouched sample. The extra complexity did not earn its place in "
                "this test.",
                icon=":material/balance:",
            )
        if trained_result.prediction_was_clipped:
            st.error(
                "The current raw ML forecast was outside the permitted 0.0001% to 500% "
                "annualized range and was clipped. Treat the current scenario as unstable.",
                icon=":material/error:",
            )

        with st.container(horizontal=True):
            st.metric(
                "Current ML volatility forecast",
                f"{trained_result.current_forecast_annualized_volatility:.1%}",
                delta=f"{current_volatility_difference:.1%} vs EWMA",
                delta_color="inverse",
                border=True,
            )
            st.metric(
                "Untouched periods tested",
                f"{evaluation.testing_samples}",
                help="Chronologically later, non-overlapping periods excluded from tuning.",
                border=True,
            )
            st.metric(
                "ML mean absolute error",
                f"{evaluation.ml_mae:.1%}",
                help=(
                    "Average absolute difference between forecast and realized annualized "
                    "volatility."
                ),
                border=True,
            )
            st.metric(
                "EWMA mean absolute error",
                f"{evaluation.ewma_mae:.1%}",
                border=True,
            )

        st.subheader("1. Did ML forecast volatility better?")
        prediction_chart = evaluation.predictions.set_index("outcome_date")[
            [
                "actual_volatility",
                "ml_predicted_volatility",
                "ewma_predicted_volatility",
            ]
        ].rename(
            columns={
                "actual_volatility": "Realized volatility",
                "ml_predicted_volatility": "Ridge ML forecast",
                "ewma_predicted_volatility": "EWMA forecast",
            }
        )
        st.line_chart(
            prediction_chart,
            y=["Realized volatility", "Ridge ML forecast", "EWMA forecast"],
            y_label="Annualized volatility",
        )
        with st.container(horizontal=True):
            st.metric(
                "Selected ridge strength",
                f"{evaluation.selected_alpha:g}",
                help="Chosen by time-series validation inside the initial training sample only.",
                border=True,
            )
            st.metric(
                "ML root mean squared error",
                f"{evaluation.ml_rmse:.1%}",
                help="Penalizes larger forecast misses more heavily than MAE.",
                border=True,
            )
            st.metric(
                "EWMA root mean squared error",
                f"{evaluation.ewma_rmse:.1%}",
                border=True,
            )
            st.metric(
                "Total non-overlapping examples",
                f"{trained_result.dataset_rows}",
                border=True,
            )
        display_predictions = evaluation.predictions.copy()
        display_predictions["forecast_date"] = pd.to_datetime(
            display_predictions["forecast_date"]
        ).dt.date
        display_predictions["outcome_date"] = pd.to_datetime(
            display_predictions["outcome_date"]
        ).dt.date
        st.dataframe(
            display_predictions,
            hide_index=True,
            width="stretch",
            column_config={
                "forecast_date": st.column_config.DateColumn("Forecast made", format="MMM D, YYYY"),
                "outcome_date": st.column_config.DateColumn(
                    "Outcome measured", format="MMM D, YYYY"
                ),
                "actual_volatility": st.column_config.NumberColumn(
                    "Realized volatility",
                    format="percent",
                ),
                "ml_predicted_volatility": st.column_config.NumberColumn(
                    "ML forecast",
                    format="percent",
                ),
                "ewma_predicted_volatility": st.column_config.NumberColumn(
                    "EWMA forecast",
                    format="percent",
                ),
                "ml_absolute_error": st.column_config.NumberColumn(
                    "ML absolute error",
                    format="percent",
                ),
                "ewma_absolute_error": st.column_config.NumberColumn(
                    "EWMA absolute error",
                    format="percent",
                ),
            },
        )

        st.subheader("2. How the ML forecast changes today's risk simulation")
        st.markdown(
            "RiskForge uses the ML forecast only as the starting volatility for the existing "
            "filtered historical simulation. Both rows reuse the same empirical shocks, seed, "
            "and EWMA path updates, isolating the effect of the starting forecast."
        )
        ml_risk_comparison = trained_result.current_risk_comparison.copy()
        ml_risk_chart = ml_risk_comparison.set_index("model")[["var_95", "expected_shortfall_95"]]
        ml_risk_chart.columns = ["95% VaR", "95% Expected Shortfall"]
        st.bar_chart(
            ml_risk_chart,
            y=["95% VaR", "95% Expected Shortfall"],
            y_label="Loss fraction",
        )
        st.dataframe(
            ml_risk_comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "model": "Starting-volatility model",
                "var_95": st.column_config.NumberColumn("95% VaR", format="percent"),
                "var_99": st.column_config.NumberColumn("99% VaR", format="percent"),
                "expected_shortfall_95": st.column_config.NumberColumn(
                    "Average worst 5% loss",
                    format="percent",
                ),
                "expected_shortfall_99": st.column_config.NumberColumn(
                    "Average worst 1% loss",
                    format="percent",
                ),
                "probability_of_loss": st.column_config.NumberColumn(
                    "Chance of loss",
                    format="percent",
                ),
                "maximum_drawdown_95": st.column_config.NumberColumn(
                    "95% drawdown cutoff",
                    format="percent",
                ),
            },
        )
        ml_fan_chart = build_fan_chart(
            trained_result.ml_price_paths,
            ticker,
            sample_paths=20,
        )
        ml_fan_chart.update_layout(
            title=(f"ML-scaled {ticker} paths over {horizon_label(trained_result.horizon)}"),
            template="plotly_dark",
            height=560,
        )
        st.plotly_chart(ml_fan_chart, width="stretch", config={"displaylogo": False})

        st.subheader("3. What the model learned")
        feature_labels = {
            "log_volatility_5": "Recent 5-day volatility",
            "log_volatility_20": "Recent 20-day volatility",
            "log_volatility_63": "Recent 63-day volatility",
            "log_ewma_volatility": "EWMA volatility",
            "absolute_return_1": "Latest absolute return",
            "downside_volatility_5": "Recent downside volatility",
            "trend_20": "Recent 20-day trend",
            "log_volatility_ratio_5_to_20": "5-day versus 20-day volatility",
        }
        coefficient_chart = trained_result.coefficients.copy()
        coefficient_chart["feature"] = coefficient_chart["feature"].replace(feature_labels)
        st.bar_chart(
            coefficient_chart,
            x="feature",
            y="standardized_coefficient",
            horizontal=True,
            x_label="Standardized ridge coefficient",
            y_label="Feature",
        )
        st.caption(
            "Positive coefficients push the log-volatility forecast higher; negative ones push "
            "it lower. Coefficients describe this fitted model and are not causal explanations."
        )

        with st.expander("Leakage controls and limitations", icon=":material/security:"):
            st.markdown(
                f"""
                - Features at each forecast date use only returns known by that date.
                - Forecast targets cover the next {horizon_label(trained_result.horizon)} and
                  never overlap.
                - The first {evaluation.initial_training_samples} examples form the initial
                  training sample. Ridge strength is selected with expanding time-series splits
                  inside that sample only.
                - Each of the following {evaluation.testing_samples} predictions is generated
                  before its outcome is added to the expanding training set.
                - The final current forecast is refit on all now-known labeled history, using
                  the already selected ridge strength.
                - Daily closing prices are a coarse volatility measure. Intraday realized
                  volatility, options-implied volatility, news, and fundamentals are absent.
                - Testing several designs against the same held-out period eventually turns it
                  into training data. A genuinely new period is needed for continued validation.
                """
            )

with tab_position:
    st.markdown(
        "**What this does:** converts one simulated loss rate into a position limit using "
        "two guardrails you choose: how much of the whole portfolio may be lost in that "
        "scenario, and how much may be concentrated in one ticker."
    )
    st.warning(
        "This calculator does not maximize gains and does not recommend a trade. It only "
        "shows the arithmetic implied by your inputs and this simulation's assumptions.",
        icon=":material/shield:",
    )
    st.caption(
        f"Risk source: the main {configured_model_label} simulation over "
        f"{horizon_label(horizon_days)}. The adaptive-volatility comparison does not silently "
        "replace this input."
    )

    default_portfolio_value = max(100_000.0, investment)
    default_current_position = min(investment, default_portfolio_value)
    with st.form("position_sizing_controls"):
        position_columns = st.columns(2)
        with position_columns[0]:
            portfolio_value_input = st.number_input(
                "Total investable portfolio value ($)",
                min_value=100.0,
                max_value=1_000_000_000.0,
                value=default_portfolio_value,
                step=1_000.0,
                help=(
                    "Use the value of the investment portfolio being protected, not income "
                    "or total net worth."
                ),
            )
            current_position_input = st.number_input(
                f"Current market value held in {ticker} ($)",
                min_value=0.0,
                max_value=1_000_000_000.0,
                value=default_current_position,
                step=500.0,
                help="Enter zero when evaluating a possible new position.",
            )
            position_risk_measure = st.selectbox(
                "Risk yardstick",
                options=list(RiskMeasure),
                index=0,
                format_func=lambda measure: measure.value,
                help=(
                    "Expected Shortfall considers the size of losses beyond VaR, while "
                    "maximum drawdown measures a path's largest temporary fall."
                ),
            )
        with position_columns[1]:
            risk_budget_percent = st.number_input(
                "Maximum portfolio loss in this modeled scenario (%)",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help=(
                    "For example, 1% means the scenario loss from this ticker should not "
                    "exceed 1% of the whole portfolio. This is your input, not a recommendation."
                ),
            )
            concentration_cap_percent = st.number_input(
                "Maximum share of portfolio in this ticker (%)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help=(
                    "This separate cap can limit a position even when its modeled loss rate "
                    "looks small."
                ),
            )
            st.caption(
                f"Current adjusted price used for unit conversion: "
                f"${float(prices.iloc[-1]):,.2f}. Fractional units are allowed in the estimate."
            )
        position_submitted = st.form_submit_button(
            "Calculate modeled position limit",
            type="primary",
            icon=":material/calculate:",
        )

    if position_submitted:
        try:
            assert isinstance(position_risk_measure, RiskMeasure)
            selected_risk_rate = risk_rate_from_summary(summary, position_risk_measure)
            sizing_result = calculate_position_limit(
                portfolio_value=float(portfolio_value_input),
                current_position=float(current_position_input),
                current_price=float(prices.iloc[-1]),
                risk_rate=selected_risk_rate,
                risk_budget_fraction=float(risk_budget_percent) / 100,
                concentration_cap_fraction=float(concentration_cap_percent) / 100,
            )
            st.session_state["riskforge_position"] = {
                "ticker": ticker,
                "result": sizing_result,
                "risk_measure": position_risk_measure.value,
            }
            st.toast("Position limit calculated", icon=":material/check_circle:")
        except (ValueError, TypeError) as exc:
            st.session_state["riskforge_position"] = None
            st.error(str(exc), icon=":material/error:")

    position_payload = st.session_state.get("riskforge_position")
    if not position_payload or position_payload["ticker"] != ticker:
        st.info(
            "Enter your own portfolio guardrails and select **Calculate modeled position "
            "limit**. RiskForge will not choose a risk budget for you.",
            icon=":material/lightbulb:",
        )
    else:
        sized_position = position_payload["result"]
        assert isinstance(sized_position, PositionSizingResult)
        selected_measure_label = str(position_payload["risk_measure"])
        selected_measure = RiskMeasure(selected_measure_label)

        if sized_position.current_status == "above_limits":
            st.error(
                f"The entered {ticker} holding is ${sized_position.amount_above_limit:,.0f} "
                "above the limit implied by these inputs. This is a model warning, not an "
                "instruction to buy or sell.",
                icon=":material/error:",
            )
        elif sized_position.current_status == "within_limits":
            st.success(
                "The entered holding is inside your selected modeled guardrails. That does "
                "not make the position safe; losses can exceed the simulated yardstick.",
                icon=":material/check_circle:",
            )
        else:
            st.info(
                "No current holding was entered, so the result is shown only as a hypothetical "
                "upper limit under your inputs.",
                icon=":material/info:",
            )

        with st.container(horizontal=True):
            st.metric(
                "Limit under chosen guardrails",
                f"${sized_position.position_limit:,.0f}",
                help="The smaller of the loss-budget and concentration limits.",
                border=True,
            )
            st.metric(
                "Share of portfolio",
                f"{sized_position.position_limit_fraction:.1%}",
                border=True,
            )
            st.metric(
                "Approximate units at limit",
                f"{sized_position.units_at_limit:,.2f}",
                help="Uses the latest adjusted price and allows fractional units.",
                border=True,
            )
            st.metric(
                "Modeled loss at limit",
                f"${sized_position.modeled_loss_at_limit:,.0f}",
                help=f"Position limit multiplied by the {selected_measure_label} rate.",
                border=True,
            )

        binding_explanations = {
            "risk_budget": (
                "The **loss budget** is binding: the modeled scenario loss reaches the "
                "maximum portfolio-dollar loss before the concentration cap is reached."
            ),
            "concentration_cap": (
                "The **concentration cap** is binding: it limits exposure before the full "
                "modeled loss budget is used."
            ),
            "both": "Both guardrails produce essentially the same position limit.",
        }
        with st.container(border=True):
            st.subheader("How RiskForge reached the limit")
            st.markdown(
                f"The selected **{selected_measure_label}** is a "
                f"**{sized_position.risk_rate:.1%} modeled loss rate** over "
                f"{horizon_label(horizon_days)}. "
                f"{RISK_MEASURE_EXPLANATIONS[selected_measure]}"
            )
            st.markdown(binding_explanations[sized_position.binding_constraint])
            st.markdown(
                "**Loss-budget limit** = portfolio loss budget ÷ modeled loss rate.  "
                "**Final limit** = smaller of the loss-budget limit and concentration cap."
            )

        limit_table = pd.DataFrame(
            {
                "guardrail": [
                    "Loss-budget limit",
                    "Concentration limit",
                    "Final modeled limit",
                    "Current entered holding",
                ],
                "position_value": [
                    sized_position.risk_budget_position_limit,
                    sized_position.concentration_position_limit,
                    sized_position.position_limit,
                    sized_position.current_position,
                ],
                "share_of_portfolio": [
                    sized_position.risk_budget_position_limit / sized_position.portfolio_value,
                    sized_position.concentration_cap_fraction,
                    sized_position.position_limit_fraction,
                    sized_position.current_position / sized_position.portfolio_value,
                ],
                "modeled_scenario_loss": [
                    sized_position.risk_budget_position_limit * sized_position.risk_rate,
                    sized_position.concentration_position_limit * sized_position.risk_rate,
                    sized_position.modeled_loss_at_limit,
                    sized_position.current_modeled_loss,
                ],
            }
        )
        st.dataframe(
            limit_table,
            hide_index=True,
            width="stretch",
            column_config={
                "guardrail": "Guardrail",
                "position_value": st.column_config.NumberColumn(
                    "Position value",
                    format="dollar",
                ),
                "share_of_portfolio": st.column_config.NumberColumn(
                    "Share of portfolio",
                    format="percent",
                ),
                "modeled_scenario_loss": st.column_config.NumberColumn(
                    "Modeled scenario loss",
                    format="dollar",
                ),
            },
        )

        with st.expander("What this limit leaves out", icon=":material/report:"):
            st.markdown(
                """
                - Losses worse than the selected VaR, Expected Shortfall, or drawdown can occur.
                - The historical simulation cannot invent risks absent from its price sample.
                - Other holdings may overlap with this ticker through funds, sectors, countries,
                  employers, or correlated assets.
                - The calculation ignores taxes, trading costs, liquidity, debt, emergency cash,
                  income stability, options, short positions, and margin or leverage.
                - Personal risk capacity depends on goals and when the money will be needed.
                """
            )
        st.caption(
            "For context, FINRA explains that concentration can amplify losses and that risk "
            "tolerance is personal. Review the whole portfolio, not this number in isolation."
        )

with tab_backtest:
    st.markdown(
        "**What this checks:** RiskForge travels backward through the available history. At "
        "each checkpoint it uses only earlier data, predicts a loss cutoff, and then checks "
        "whether the following period actually crossed that cutoff."
    )
    if bootstrap_method is BootstrapMethod.MOVING_BLOCK:
        backtest_model_text = f"Moving blocks keep up to {block_size} consecutive days together."
    else:
        backtest_model_text = "The independent-day baseline samples each day separately."
    st.caption(
        f"The check uses the same selected resampling model. {backtest_model_text} Forecast "
        "periods do not overlap, which makes violations easier to interpret, but longer "
        "horizons produce fewer observations."
    )

    with st.form("backtest_controls"):
        backtest_columns = st.columns(3)
        with backtest_columns[0]:
            backtest_horizon_choice = st.selectbox(
                "Forecast horizon",
                options=list(HORIZONS),
                index=0,
                help="One-day tests provide more observations; longer horizons test longer risks.",
            )
            backtest_confidence_label = st.selectbox(
                "Loss cutoff to test",
                options=["95% VaR", "99% VaR"],
                help="A calibrated 95% VaR should be exceeded about 5 times in 100 periods.",
            )
        with backtest_columns[1]:
            backtest_training_choice = st.selectbox(
                "Training history at each checkpoint",
                options=list(HISTORY_WINDOWS),
                index=1,
                help="Only observations before each forecast are used for training.",
            )
            backtest_testing_choice = st.selectbox(
                "Past period to evaluate",
                options=list(BACKTEST_WINDOWS),
                index=1,
                help="A longer evaluation period usually gives a more informative test.",
            )
        with backtest_columns[2]:
            backtest_paths = st.select_slider(
                "Simulations at each checkpoint",
                options=[1_000, 2_000, 5_000, 10_000],
                value=2_000,
            )
            backtest_seed = st.number_input(
                "Backtest reproducibility number",
                min_value=0,
                value=42,
                step=1,
            )
        backtest_submitted = st.form_submit_button(
            "Run historical model check",
            type="primary",
            icon=":material/fact_check:",
        )

    if backtest_submitted:
        confidence = 0.95 if backtest_confidence_label == "95% VaR" else 0.99
        try:
            with st.skeleton(height=180):
                backtest_result = run_backtest(
                    ticker=ticker,
                    training_window=HISTORY_WINDOWS[str(backtest_training_choice)],
                    testing_window=BACKTEST_WINDOWS[str(backtest_testing_choice)],
                    horizon=HORIZONS[str(backtest_horizon_choice)],
                    paths=int(backtest_paths),
                    confidence=confidence,
                    seed=int(backtest_seed),
                    method=bootstrap_method.value,
                    block_size=block_size,
                )
                st.session_state["riskforge_backtest"] = {
                    "ticker": ticker,
                    "result": backtest_result,
                    "horizon": str(backtest_horizon_choice),
                    "training_window": str(backtest_training_choice),
                    "testing_window": str(backtest_testing_choice),
                    "method": bootstrap_method.value,
                    "block_size": block_size,
                }
            st.toast("Historical model check complete", icon=":material/check_circle:")
        except (MarketDataError, ValueError, TypeError) as exc:
            st.session_state["riskforge_backtest"] = None
            st.error(str(exc), icon=":material/error:")
        except Exception as exc:
            st.session_state["riskforge_backtest"] = None
            st.error(f"The historical model check failed: {exc}", icon=":material/error:")

    backtest_payload = st.session_state.get("riskforge_backtest")
    if not backtest_payload or backtest_payload["ticker"] != ticker:
        st.info(
            "Select **Run historical model check** to see whether this model's loss cutoffs "
            "matched past outcomes.",
            icon=":material/lightbulb:",
        )
    else:
        checked_result = backtest_payload["result"]
        assert isinstance(checked_result, BacktestResult)
        show_backtest_status(checked_result)
        st.caption(
            "A backtest can reveal weaknesses in a model, but passing one never proves that "
            "future losses will be contained."
        )

        expected_per_100 = round(checked_result.expected_violation_rate * 100)
        observed_per_100 = round(checked_result.observed_violation_rate * 100)
        with st.container(border=True):
            st.subheader("The result in everyday language")
            st.markdown(
                f"RiskForge tested **{checked_result.observations} non-overlapping periods**. "
                f"A {checked_result.confidence:.0%} loss cutoff should be crossed about "
                f"**{expected_per_100} times in 100** if its coverage is well calibrated. "
                f"The actual result was **{observed_per_100} times in 100** "
                f"({checked_result.violations} violations)."
            )

        with st.container(horizontal=True):
            st.metric("Periods tested", f"{checked_result.observations:,}", border=True)
            st.metric(
                "Actual violations",
                f"{checked_result.violations}",
                help="Times the realized loss was larger than the model's predicted cutoff.",
                border=True,
            )
            st.metric(
                "Expected violations",
                f"{checked_result.expected_violations:.1f}",
                help="The long-run count implied by the selected confidence level.",
                border=True,
            )
            st.metric(
                "Observed violation rate",
                f"{checked_result.observed_violation_rate:.1%}",
                help=(
                    "The 95% uncertainty interval is "
                    f"{checked_result.violation_rate_ci_low:.1%} to "
                    f"{checked_result.violation_rate_ci_high:.1%}."
                ),
                border=True,
            )

        with st.container(border=True):
            st.subheader("Do the misses bunch together?")
            st.markdown(
                "A useful risk model should not only miss roughly the expected number of "
                "times; those misses should not arrive in runs. Clusters can signal that "
                "market conditions changed faster than the model adapted."
            )
            show_independence_status(checked_result)
            with st.container(horizontal=True):
                st.metric(
                    "Violation after a safe period",
                    format_optional_rate(checked_result.violation_after_safe_rate),
                    help="How often the next forecast missed after the previous one was safe.",
                    border=True,
                )
                st.metric(
                    "Violation after a violation",
                    format_optional_rate(checked_result.violation_after_violation_rate),
                    help=(
                        "How often another miss immediately followed a miss. A materially "
                        "higher rate can indicate clustering."
                    ),
                    border=True,
                )
                st.metric(
                    "Longest run of violations",
                    f"{checked_result.max_consecutive_violations}",
                    help="The largest number of consecutive periods in which VaR was exceeded.",
                    border=True,
                )
        st.plotly_chart(
            build_backtest_loss_chart(checked_result),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "A red X marks a violation: the loss that occurred was larger than the cutoff the "
            "model predicted using only information available at that checkpoint."
        )
        st.plotly_chart(
            build_cumulative_coverage_chart(checked_result),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "The shaded region is a 95% binomial reference range. Moving outside it is a "
            "warning that violations may be accumulating unusually quickly or slowly."
        )

        with st.expander("Statistical details", icon=":material/functions:"):
            st.markdown(
                f"""
                **Kupiec unconditional-coverage test**

                - Test statistic: `{checked_result.kupiec_statistic:.3f}`
                - p-value: `{checked_result.kupiec_p_value:.3f}`
                - Observed violation-rate interval:
                  `{checked_result.violation_rate_ci_low:.1%}` to
                  `{checked_result.violation_rate_ci_high:.1%}`

                A p-value below 0.05 flags evidence that the overall violation rate differs
                from the model's claim. It is **not** the probability that the model is correct.

                **Christoffersen independence test**

                - Test statistic: `{checked_result.independence_statistic:.3f}`
                - p-value: `{checked_result.independence_p_value:.3f}`

                This checks whether a violation is more or less likely after the previous
                period also violated its cutoff. A p-value below 0.05 flags dependence; a
                higher post-violation rate points specifically to clustering.

                **Combined conditional-coverage test**

                - Test statistic: `{checked_result.conditional_coverage_statistic:.3f}`
                - p-value: `{checked_result.conditional_coverage_p_value:.3f}`

                This combines the overall-rate and independence checks. These likelihood-ratio
                tests rely on large-sample approximations, so few violations mean limited power.
                """
            )

            st.markdown("**Transition counts used by the independence test**")
            transition_counts = pd.DataFrame(
                {
                    "Next period: safe": [
                        checked_result.transition_00,
                        checked_result.transition_10,
                    ],
                    "Next period: violation": [
                        checked_result.transition_01,
                        checked_result.transition_11,
                    ],
                },
                index=["Previous period: safe", "Previous period: violation"],
            )
            st.dataframe(transition_counts, width="stretch")

            display_forecasts = checked_result.forecasts[
                ["outcome_date", "predicted_var", "realized_loss", "violation"]
            ].copy()
            display_forecasts["outcome_date"] = pd.to_datetime(
                display_forecasts["outcome_date"]
            ).dt.date
            st.dataframe(
                display_forecasts,
                hide_index=True,
                width="stretch",
                column_config={
                    "outcome_date": st.column_config.DateColumn(
                        "Outcome date",
                        format="MMM D, YYYY",
                    ),
                    "predicted_var": st.column_config.NumberColumn(
                        "Predicted loss cutoff",
                        format="percent",
                    ),
                    "realized_loss": st.column_config.NumberColumn(
                        "Loss that occurred",
                        format="percent",
                    ),
                    "violation": "Cutoff exceeded?",
                },
            )

with tab_audit:
    single_snapshot = create_single_asset_audit(result, quality=quality)
    safe_ticker = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in ticker
    ).strip("._")
    render_audit_panel(
        single_snapshot,
        quality,
        file_stem=f"riskforge_{safe_ticker or 'ticker'}",
    )
    st.info(
        "The existing terminal-scenarios CSV contains path endings. The audit pack serves a "
        "different purpose: it preserves the settings, data checks, metrics, and caveats "
        "needed to interpret those outcomes.",
        icon=":material/info:",
    )

with tab_details:
    if bootstrap_method is BootstrapMethod.MOVING_BLOCK:
        sampling_step = (
            f"Randomly select {min(block_size, horizon_days)}-day consecutive historical "
            "blocks, join them, and reuse blocks as needed."
        )
        dependence_limit = (
            f"It preserves dependence only inside fixed {min(block_size, horizon_days)}-day "
            "blocks; relationships can break at block boundaries."
        )
    else:
        sampling_step = "Randomly sample individual historical returns with replacement."
        dependence_limit = (
            "It treats daily returns as independent, so it can miss volatility clustering."
        )

    labels = {
        "var_95": ("95% Value at Risk", "A loss cutoff exceeded in about 5% of simulations."),
        "var_99": ("99% Value at Risk", "A loss cutoff exceeded in about 1% of simulations."),
        "expected_shortfall_95": (
            "95% Expected Shortfall",
            "The average loss among the worst 5% of simulations.",
        ),
        "expected_shortfall_99": (
            "99% Expected Shortfall",
            "The average loss among the worst 1% of simulations.",
        ),
        "probability_of_loss": (
            "Probability of any loss",
            "The share of simulations that ended below the starting price.",
        ),
        "probability_exceeding_threshold": (
            f"Probability of losing {threshold:.0%} or more",
            "The share that crossed the loss level you chose.",
        ),
        "mean_maximum_drawdown": (
            "Average maximum drawdown",
            "The average largest temporary fall from a previous high within each path.",
        ),
        "maximum_drawdown_95": (
            "95th percentile maximum drawdown",
            "Only about 5% of paths had a larger peak-to-trough fall.",
        ),
    }
    details = pd.DataFrame(
        [
            {
                "Measure": labels[key][0],
                "Result": percent(value),
                "What it means": labels[key][1],
            }
            for key, value in asdict(summary).items()
        ]
    )
    st.dataframe(details, hide_index=True, width="stretch")

    with st.expander("How RiskForge creates the simulation", icon=":material/schema:"):
        st.markdown(
            f"""
            1. Download adjusted daily closing prices for the selected ticker.
            2. Convert price changes into daily percentage-like log returns.
            3. {sampling_step}
            4. Chain the sampled returns into thousands of possible price paths.
            5. Count how often losses and drawdowns occur across those paths.

            Keeping the same **reproducibility number** produces the same random sample, which
            makes experiments easier to compare.
            """
        )

    with st.expander("What this model cannot tell you", icon=":material/report:"):
        st.markdown(
            f"""
            - It cannot predict earnings, interest rates, wars, crashes, or tomorrow's news.
            - It cannot create a type of event that never appeared in the chosen history.
            - {dependence_limit}
            - A fixed block model does not automatically recognize or forecast changing
              market regimes.
            - It ignores fees, taxes, liquidity, trading costs, and your personal finances.
            - A low simulated probability does not mean an event is impossible.
            """
        )

st.subheader("A useful next experiment", icon=":material/science:")
st.markdown(
    "Run the **ML lab** on a long history and record whether ridge regression actually beats "
    "EWMA. Then test the same locked design on a different ticker or genuinely later period. "
    "A model that wins only where it was repeatedly inspected is probably overfit."
)

terminal_scenarios = pd.DataFrame(
    {
        "path": np.arange(1, price_paths.shape[0] + 1),
        "terminal_price": price_paths[:, -1],
        "terminal_return": terminal_returns(price_paths),
    }
)
st.download_button(
    "Download the simulated outcomes",
    data=terminal_scenarios.to_csv(index=False),
    file_name=f"riskforge_{ticker}_terminal_scenarios.csv",
    mime="text/csv",
    on_click="ignore",
    icon=":material/download:",
    help="Downloads one row for each simulated future as a CSV file.",
    width="content",
)
st.subheader("Export this analysis", icon=":material/download:")
st.caption(
    "Save a readable report containing the model settings, data checks, risk results, and "
    "limitations."
)
st.download_button(
    "Export analysis report",
    data=audit_html(single_snapshot),
    file_name=f"riskforge_{safe_ticker or 'ticker'}_analysis_report.html",
    mime="text/html",
    on_click="ignore",
    type="primary",
    icon=":material/download:",
    width="stretch",
    key="single_bottom_report_export",
)
