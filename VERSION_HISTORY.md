# RiskForge version history and design rationale

This document explains how RiskForge grew from a single-ticker Monte Carlo prototype into
the current market-risk research application. It records both the edits and the reason
behind them so that readers can evaluate the modeling decisions instead of treating the
application as a black box.

## Historical accuracy

Most features were developed before the repository received its first Git commit. The
development stages below reconstruct that work from the implemented modules, tests, and
project requirements; they are milestones, not fabricated Git releases. The published Git
history begins at `v0.1.0`.

## Published releases

| Version | Main edit | Reason |
| --- | --- | --- |
| `v0.1.0` | Captured the complete initial application, tests, documentation, CI, and reproducible lockfile. | Establish an honest baseline instead of manufacturing a commit for every earlier development prompt. |
| `v0.1.1` | Removed ANSI terminal styling before checking Typer help text in the CLI test. | GitHub Actions enables colored output on Linux, causing a literal help-text assertion that passed on Windows to fail in CI. The application behavior was correct; the test needed to be platform-independent. |
| `v0.1.2` | Added this rationale-based history and linked it from the README. | A concise changelog says what shipped, but reviewers and learners also need to know why each modeling and UX decision was made. |

## Development milestones

### 1. Reproducible Python foundation

**Edits**

- Created a Python 3.12 project using a `src/` package layout.
- Separated data access, return calculation, simulation, metrics, visualization, and the
  CLI into focused modules.
- Added a locked `uv` environment, pytest, Ruff, and type hints.

**Reason**

Financial experiments become difficult to trust when calculations, downloaded data, and
interface code are mixed together. The package boundaries make individual assumptions
testable, while the lockfile and fixed Python version reduce machine-to-machine drift.

**Main files:** `pyproject.toml`, `uv.lock`, `src/riskforge/`, `tests/`

### 2. Market data and return preparation

**Edits**

- Added adjusted daily price downloads through Yahoo Finance for arbitrary ticker symbols.
- Normalized downloaded data and calculated continuously compounded log returns.
- Added validation and human-readable errors for empty, malformed, or insufficient data.

**Reason**

Adjusted prices account for corporate actions more appropriately than unadjusted closes.
Log returns add across time, which makes simulated multi-day paths straightforward. Any
ticker support makes the engine reusable, but stronger validation is necessary because
symbols, listing history, and provider responses vary.

**Main files:** `data.py`, `returns.py`, `quality.py`

### 3. Deterministic historical-bootstrap MVP

**Edits**

- Added IID sampling from observed daily log returns.
- Added explicit NumPy seeds and deterministic shape validation.
- Converted sampled returns back into price paths from the latest adjusted price.

**Reason**

Historical bootstrapping is transparent: simulated days come from returns that actually
occurred. A fixed seed makes a result reproducible for testing and review. IID sampling was
chosen as the understandable baseline, while recognizing that it ignores serial dependence
and changing volatility.

**Main files:** `simulation.py`, `returns.py`

### 4. Downside-focused risk measurements

**Edits**

- Added 95% and 99% Value at Risk and Expected Shortfall.
- Added probability of loss and probability of crossing a user-defined loss threshold.
- Added path-level maximum drawdown summaries.

**Reason**

A fan chart alone does not answer practical risk questions. VaR provides a loss quantile,
Expected Shortfall describes losses beyond that quantile, threshold probability supports a
user-defined concern, and drawdown captures painful path behavior that terminal returns can
hide. Multiple metrics are shown because no single number is a complete risk assessment.

**Main files:** `metrics.py`, `outcomes.py`

### 5. Command-line analysis and visualization

**Edits**

- Added the `riskforge simulate TICKER` Typer command.
- Added Rich summary tables and interactive Plotly fan charts.
- Added input bounds, safe output filenames, and clear exceptions.

**Reason**

The CLI provides a reproducible interface for scripts and technical users. Tables make the
core outputs scannable, while percentile bands and sample paths show the distribution rather
than presenting a single forecast as certain.

**Main files:** `cli.py`, `visualization.py`

### 6. Interactive Streamlit dashboard

**Edits**

- Added browser-based controls for ticker, horizon, path count, lookback, seed, model, and
  loss threshold.
- Added interactive result cards, charts, tabs, and downloads.
- Added deployment configuration for an externally hosted Streamlit application.

**Reason**

The CLI is efficient but assumes terminal familiarity. A guided dashboard lets a first-time
user explore assumptions and see how changing an input affects modeled risk without editing
code.

**Main files:** `streamlit_app.py`, `.streamlit/config.toml`

### 7. Plain-language education and safer wording

**Edits**

- Added onboarding text, metric explanations, assumptions, limitations, and tooltips.
- Added explicit statements that results are educational and not investment advice.
- Presented positive outcomes as green bullish scenarios and negative outcomes as red
  bearish scenarios, rather than labeling an investment as safe or unsafe.

**Reason**

Probability estimates can create false confidence when users do not understand sampling
risk or model limitations. Everyday language reduces misinterpretation. Bullish/bearish
describes the simulated direction; safe/unsafe would incorrectly imply a recommendation or
guarantee.

**Main files:** `streamlit_app.py`, `outcomes.py`, `README.md`

### 8. Moving-block bootstrap

**Edits**

- Added moving-block resampling alongside IID sampling.
- Added a configurable block length and made the block model available in the CLI and UI.

**Reason**

Daily returns are not always independent. Sampling consecutive historical blocks retains
some short-run clustering and sequence structure. Keeping IID as a baseline lets users see
whether this assumption materially changes their result.

**Main files:** `simulation.py`, `cli.py`, `streamlit_app.py`

### 9. Historical stress and regime context

**Edits**

- Added rolling realized outcomes from historical windows comparable to the selected
  horizon.
- Added calm/turbulent and advancing/declining regime descriptions.
- Compared simulated losses with actual historical stress periods.

**Reason**

A simulation should be checked against events that really occurred. Regime context also
helps explain why a recent sample may not represent all market conditions. These labels are
descriptive context, not market-timing signals.

**Main files:** `stress.py`, `regimes.py`

### 10. Walk-forward VaR backtesting

**Edits**

- Added rolling out-of-sample VaR forecasts and violation tracking.
- Added coverage, independence, and conditional-coverage diagnostics.

**Reason**

A risk model should not be judged only by how plausible its chart looks. Backtesting asks
whether realized losses breached forecast VaR at a rate and pattern consistent with the
claimed confidence level. It can reveal systematic underestimation or clustered failures.

**Main files:** `backtesting.py`

### 11. Model robustness comparisons

**Edits**

- Added comparisons across IID and moving-block sampling and multiple lookback windows.
- Displayed metric ranges and model sensitivity rather than only one selected result.

**Reason**

Monte Carlo outputs depend on modeling choices. Showing how conclusions move when the model
or history changes makes model risk visible and discourages reliance on a convenient single
number.

**Main files:** `robustness.py`

### 12. Risk-budget position sizing

**Edits**

- Added illustrative position limits using VaR, Expected Shortfall, drawdown, and
  concentration caps.
- Reported the binding constraint and preserved explicit non-recommendation language.

**Reason**

Percentage risk becomes more concrete when translated into dollars at risk. Using several
caps avoids pretending one metric is sufficient, while calling the result a guardrail rather
than an optimal trade keeps the output within the model's actual evidence.

**Main files:** `position_sizing.py`

### 13. Volatility-aware historical simulation

**Edits**

- Added EWMA volatility estimation and filtered historical simulation.
- Standardized historical shocks and rescaled them to current estimated volatility.
- Added walk-forward comparisons with simpler models.

**Reason**

Raw historical sampling gives a quiet day from years ago the same scale it originally had,
even when current volatility is very different. Filtering retains empirical shock shapes
while adapting their scale. The simpler bootstrap remains visible because volatility filters
introduce their own decay-rate assumptions.

**Main files:** `volatility.py`

### 14. Transparent ML volatility lab

**Edits**

- Added ridge-regression volatility forecasts using explicit lagged features.
- Added chronological nested validation, coefficient inspection, and an EWMA benchmark.
- Kept the ML model out of the core price-direction forecast.

**Reason**

This adds machine-learning practice without turning the app into an opaque stock picker.
Chronological validation reduces leakage, regularization stabilizes correlated features, and
the benchmark shows whether added complexity earns its place.

**Main files:** `ml_volatility.py`

### 15. Joint portfolio-risk simulation

**Edits**

- Added two-to-ten-asset joint return resampling and buy-and-hold weight drift.
- Added correlations, standalone-versus-portfolio risk, diversification gaps, and Expected
  Shortfall tail contributions.

**Reason**

Adding separate single-stock VaRs misses diversification and simultaneous losses. Sampling
the same historical row or block across assets preserves observed cross-asset dependence.
Tail contributions identify which holdings dominate the worst portfolio scenarios.

**Main files:** `portfolio.py`

### 16. Factor attribution and factor stress

**Edits**

- Added proxy-factor betas, explanatory power, residual risk, VIF, and matrix-conditioning
  diagnostics.
- Added custom linear factor shocks and estimated portfolio impacts.

**Reason**

Ticker count is not the same as economic diversification. Factor exposure can reveal that
different holdings share the same market, sector, rate, or style risk. Collinearity warnings
make unstable beta estimates visible instead of presenting them as precise facts.

**Main files:** `factor_attribution.py`

### 17. Constrained allocation sandbox

**Edits**

- Added long-only allocation experiments with maximum-weight, turnover, Expected Shortfall,
  and drawdown constraints.
- Added minimum-volatility, minimum-Expected-Shortfall, and risk-adjusted objectives.
- Added covariance shrinkage and multi-start optimization.

**Reason**

Unconstrained optimizers often produce concentrated and unstable weights because estimated
returns and covariances are noisy. Constraints and shrinkage make the experiment more
realistic, while the sandbox label makes clear that scenario feasibility is not a guarantee
of future safety.

**Main files:** `allocation.py`

### 18. Portfolio shopping-cart interaction

**Edits**

- Added drag-and-drop holdings, accessible arrow controls, and a dollar investment field for
  each ticker.
- Converted dollar allocations into weights automatically.
- Invalidated old results whenever the cart changed.

**Reason**

Most new users think in investment dollars before portfolio weights. The shopping-cart
metaphor makes allocation approachable, while stale-result invalidation prevents a chart
from silently representing holdings that are no longer selected.

**Main files:** `streamlit_components.py`, `streamlit_app.py`

### 19. Data quality, auditability, and exports

**Edits**

- Added gates for ordering, duplicates, finite and positive prices, history length,
  staleness, variation, and extreme moves.
- Added deterministic audit fingerprints and downloadable JSON and HTML reports.
- Added a bottom-of-analysis complete-report export.

**Reason**

Sophisticated calculations cannot rescue invalid inputs. The quality gate stops or warns
before unreliable data reaches the model. Audit fingerprints, inputs, and reports make a
run easier to reproduce, review, and discuss without relying on a screenshot.

**Main files:** `quality.py`, `reporting.py`, `streamlit_app.py`

### 20. Deployment and continuous verification

**Edits**

- Added `riskforge health` for offline runtime, dependency, seed, metadata, entrypoint, and
  lockfile checks.
- Added GitHub Actions for locked installation, formatting, linting, tests, and health.
- Ignored local environments, caches, generated reports, and Streamlit secrets.

**Reason**

A project intended for GitHub and external deployment needs repeatable checks outside the
developer's computer. CI caught a real Windows/Linux difference in ANSI-styled CLI help,
which led directly to the `v0.1.1` portability fix.

**Main files:** `health.py`, `.github/workflows/ci.yml`, `.gitignore`

## Continuing principles

- Treat every output as a conditional model estimate, not a promise or recommendation.
- Prefer transparent baselines before adding complex models.
- Use chronological validation for time-series forecasts.
- Show model sensitivity and failure modes alongside headline results.
- Keep deterministic seeds and record enough inputs to reproduce a run.
- Separate calculation code from interface code and test both in proportion to risk.
- Add portfolio, ML, or optimization features only when their assumptions can be explained
  in plain language.
