# Changelog

All notable changes to RiskForge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Continue validating model behavior and improving the educational explanations.
- Treat new models and interfaces as research features until they have adequate tests.

## [0.1.0] - 2026-09-28

### Initial market-risk engine

- Added a Python 3.12 project with a `src/` package layout.
- Added adjusted daily price downloads for arbitrary Yahoo Finance ticker symbols.
- Added log-return calculation and deterministic NumPy random seeds.
- Added IID historical-bootstrap simulation and price-path conversion.
- Added 95% and 99% Value at Risk and Expected Shortfall, probability of loss,
  user-defined loss-threshold probability, and maximum drawdown.
- Added a Typer command-line interface, Rich result tables, and interactive Plotly
  fan-chart export.

### Interactive education

- Added an interactive Streamlit dashboard for single securities and portfolios.
- Added first-time-user guidance, plain-language metric explanations, model assumptions,
  limitations, and prominent educational-use warnings.
- Added ticker examples, input validation, and clear data-download error messages.
- Added green bullish and red bearish modeled-outcome treatments without describing a
  simulation as a guaranteed safe or unsafe investment.

### Simulation and stress analysis

- Added a moving-block bootstrap that preserves short historical return sequences.
- Added EWMA-filtered historical simulation for volatility-aware empirical shocks.
- Added historical rolling stress comparisons and calm/turbulent,
  advancing/declining regime context.
- Added systematic robustness comparisons across models and historical lookback windows.

### Validation and volatility research

- Added walk-forward VaR backtesting with coverage, independence, and conditional-coverage
  diagnostics.
- Added a transparent ridge-regression volatility lab with explicit features,
  chronological nested validation, coefficients, and an EWMA benchmark.
- Kept the ML output as a research comparison rather than a trading signal.

### Portfolio and allocation research

- Added joint multi-asset bootstrap simulation with buy-and-hold weight drift.
- Added correlation analysis, diversification gaps, standalone risk, and Expected
  Shortfall tail contributions.
- Added proxy-factor attribution, beta estimates, explanatory power, residual risk,
  multicollinearity diagnostics, and custom factor stress scenarios.
- Added an allocation sandbox with long-only, maximum-weight, turnover, Expected
  Shortfall, and drawdown constraints; covariance shrinkage; multiple objectives; and
  multi-start optimization.
- Added illustrative risk-budget position-sizing guardrails using VaR, Expected
  Shortfall, drawdown, and concentration caps.
- Added a drag-and-drop portfolio shopping cart with per-ticker dollar allocations,
  automatic weights, accessible arrow controls, and stale-result invalidation.

### Reproducibility and operations

- Added data-quality gates for ordering, duplicates, non-finite or non-positive values,
  insufficient history, stale data, and extreme moves.
- Added deterministic audit fingerprints and JSON and HTML analysis-report downloads.
- Added the `riskforge health` offline deployment check.
- Added pytest coverage for the engine and application support modules.
- Added Ruff linting and formatting configuration and a GitHub Actions quality gate.
- Excluded virtual environments, caches, local secrets, and generated HTML reports from
  version control.

### Important limitations

- RiskForge is an educational and research tool, not financial advice.
- Historical resampling cannot predict structural breaks or guarantee future outcomes.
- Results can be materially affected by data quality, model selection, lookback windows,
  liquidity, fees, taxes, slippage, and market impact.
- The project does not yet include GARCH, brokerage integration, trade execution, or a
  production portfolio-management workflow.

[Unreleased]: https://github.com/gunhoo1124-arch/riskforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gunhoo1124-arch/riskforge/releases/tag/v0.1.0
