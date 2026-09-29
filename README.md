# RiskForge

RiskForge is an educational Python 3.12 market-risk engine. It downloads adjusted
daily prices for Yahoo Finance tickers, resamples historical log returns, produces
Monte Carlo paths, and summarizes terminal-loss and drawdown distributions for a single
investment or a long-only portfolio.

It includes both a command-line interface and an interactive Streamlit dashboard.

RiskForge implements two transparent historical resampling models: a moving-block
bootstrap that keeps short return sequences together, and an IID bootstrap baseline
that samples days independently. It also includes a joint multi-asset bootstrap and an
optional EWMA-filtered historical simulation that adapts empirical shocks to changing
volatility. It does not use black-box machine learning or GARCH. Its optional ML lab uses
regularized linear regression with explicit features and nested chronological validation.

## Installation

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository, and create the locked environment:

```powershell
git clone https://github.com/gunhoo1124-arch/riskforge.git
cd riskforge
uv sync --dev
```

For development in an existing checkout, `uv sync --dev` is sufficient.

## Usage

Run the default 20-day, 10,000-path SPY simulation:

```powershell
uv run riskforge simulate SPY --horizon 20 --paths 10000 --lookback 1000 --seed 42
```

The default uses five-day moving blocks. Compare it with the independent-day baseline:

```powershell
uv run riskforge simulate SPY --model iid
uv run riskforge simulate SPY --model moving-block --block-size 10
```

Choose a loss threshold and chart destination:

```powershell
uv run riskforge simulate AAPL --loss-threshold 0.15 --output reports/aapl.html
```

The command prints a Rich summary table and writes a self-contained interactive
Plotly HTML fan chart. Run `uv run riskforge simulate --help` for all options.

Run the offline deployment check before publishing or after changing dependencies:

```powershell
uv run riskforge health
```

This verifies the Python runtime, critical imports, deterministic NumPy seeding, package
metadata, Streamlit entrypoint, and locked environment without relying on Yahoo Finance.

## Interactive dashboard

Launch the Streamlit app locally from the repository root:

```powershell
uv run streamlit run streamlit_app.py
```

If `uv` is not available on your terminal `PATH`, use the project environment directly:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

The dashboard provides interactive controls, risk-metric cards, scenario and
distribution charts, historical-price context, downloadable terminal scenarios, and
a historical stress lab, market-regime context, systematic model comparisons,
risk-budget position limits, multi-asset factor and stress attribution, a constrained
allocation sandbox, a volatility-aware forecasting lab, walk-forward VaR backtests, and a
transparent machine-learning volatility benchmark. Each completed base simulation also has
an **Audit and export** tab with automated data-quality checks and compact JSON and HTML
reports, plus a one-click complete-report export at the bottom. Results separate positive
simulated endings in green from negative simulated endings in red. These are bullish and
bearish model scenarios, not safe/unsafe investment verdicts.

Choose **Portfolio** near the top of the dashboard to analyze two to ten holdings together.
The single-investment lab and CLI continue to work independently.

### Multi-asset portfolio risk

The **Portfolio** lab has a draggable holdings cart for any two to ten Yahoo Finance tickers.
Users enter how many hypothetical dollars they would allocate to each holding, reorder cards
by dragging or with accessible arrow buttons, and see the implied starting weights. RiskForge
normalizes those positive dollar amounts to 100%. It then downloads adjusted prices, retains
dates shared by every holding, and resamples each complete cross-asset return row as one market
observation. The moving-block option also keeps short sequences of those joint rows together.
This is deliberate: sampling each asset independently would erase observed co-movement and
could create unrealistic diversification.

The simulation uses a buy-and-hold convention. If \(w_i\) is asset \(i\)'s starting weight,
\(V_0\) is initial portfolio value, and \(r^*_{i,t}\) is its simulated log return, then:

```text
V_t* = sum_i [V_0 w_i exp(sum_(s=1)^t r*_(i,s))]
```

Weights are applied at day zero and then drift; there is no automatic rebalancing. The lab
reports portfolio VaR, Expected Shortfall, loss probability, maximum drawdown, historical
correlations, inverse-concentration effective holdings, standalone asset risk, and each
holding's average contribution inside the portfolio's worst 5% of simulated endings. Those
tail contributions add to portfolio 95% Expected Shortfall.

The displayed diversification gap is the weighted average of standalone asset risk minus
portfolio risk under the same scenarios. It is a conditional comparison, not a promised
benefit. Correlations can shift in stress, and supervisory risk guidance explicitly treats
correlation shifts, concentration, illiquidity, and stress testing as risks that can sit
outside a VaR summary. See the
[Basel Committee market-risk guidance](https://www.bis.org/committees/bcbs/basel-framework/standard/srp/20/inforce/2019-12-15/published/2019-12-15)
and its
[market-risk framework summary](https://www.bis.org/publications/fsi-summary-revised-market-risk-framework-executive-summary).

### Factor and stress attribution

After running a portfolio simulation, open **Factor and stress lab**. Choose one to six
traded ticker proxies and give each a plain-language meaning. The defaults use SPY for broad
U.S. equities, TLT for long U.S. Treasury prices, and GLD for gold, but they are examples—not
a universal factor model. A user can instead select sector ETFs, international markets,
currencies, commodities, or other proxies relevant to the portfolio.

For holding return \(r_{i,t}\) and proxy-return vector \(f_t\), RiskForge fits an ordinary
least-squares regression with an intercept:

```text
r_(i,t) = alpha_i + beta_i' f_t + epsilon_(i,t)
portfolio beta = sum_i weight_i * beta_i
```

All series use aligned daily log returns. The dashboard reports holding and portfolio betas,
in-sample R-squared, adjusted R-squared, unexplained annualized volatility, factor
correlations, variance inflation factors, and the condition number of the standardized factor
matrix. Perfectly redundant factors are rejected. High proxy overlap produces an explicit
warning because individual coefficients may be unstable.

The custom shock calculation is a linear sensitivity analysis:

```text
factor-implied portfolio return = sum_f portfolio_beta_f * entered_proxy_shock_f
```

Factor contributions and starting-weight holding contributions reconcile to the same linear
estimate. A bond-ETF input is a **price-return shock to that proxy**, not a change in interest
rates measured in percentage points. The calculation deliberately excludes the regression
residual, changing betas, nonlinear payoffs, liquidity effects, and second-order interactions,
so it is not a forecast or a complete stress test.

This cautious presentation follows the broader principle that stress scenarios should make
their assumptions and limitations explicit and cover material risks. See the
[Basel stress-testing principles](https://www.bis.org/publications/201810-guidelines-stress-testing-principles)
and the Federal Reserve's current
[model-risk guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm).

### Constrained allocation sandbox

Open **Allocation sandbox** after running a portfolio simulation. This tool compares the
current weights with one long-only candidate under explicit constraints. It is designed to
make trade-offs visible, not to prescribe trades.

Available objectives are:

- **Lowest historical volatility:** minimizes \(w'\Sigma w\), the classical covariance-based
  portfolio-risk quantity associated with
  [Markowitz portfolio selection](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).
- **Lowest simulated tail loss:** minimizes empirical 95% Expected Shortfall across one fixed
  joint-bootstrap scenario set. Expected Shortfall optimization follows the scenario-based
  risk-measure tradition developed by
  [Rockafellar and Uryasev](https://doi.org/10.1016/S0378-4266(02)00271-6).
- **Highest shrunk return per unit of volatility:** divides a shrunk historical annualized
  return estimate by historical annualized volatility. This is the most estimation-sensitive
  objective and receives the strongest dashboard warning.

The sandbox can enforce:

- weights totaling 100% with no short positions;
- a maximum weight for any holding;
- maximum one-way turnover,
  `0.5 * sum(abs(candidate_weight - current_weight))`;
- a maximum simulated 95% Expected Shortfall; and
- a maximum 95th-percentile simulated path drawdown.

Every candidate is evaluated on the same deterministically sampled asset-return paths, so
random scenario differences do not favor one weight vector. The solver uses multiple starting
points and accepts a result only after independently checking every entered constraint. If the
position and turnover limits are incompatible, or no feasible risk-capped candidate is found,
RiskForge returns a clear error rather than silently weakening a guardrail.

Historical mean returns are shrunk toward the cross-asset average before the risk-adjusted
score is calculated. Shrinkage limits extreme sample estimates but does not make them
predictive. The sandbox remains in-sample, and neither its weights nor a successful constraint
check establish out-of-sample performance. This is consistent with current supervisory
guidance that model outputs depend on assumptions, data quality, intended use, and ongoing
challenge. See the Federal Reserve's
[model-risk guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm).

### Data-quality gate and audit exports

Before a new single-asset or portfolio simulation runs, RiskForge checks that the adjusted
price table is chronological, unique, finite, positive, long enough, and non-constant.
Blocking failures stop the calculation. Stale observations and unusually large daily moves
remain visible warnings because either may be legitimate for a particular market but still
deserves review.

The **Audit and export** tab shows every check and creates two compact reports:

- JSON for structured storage or later comparison; and
- standalone HTML for a readable local record.

Both record the model inputs, NumPy seed, data window, price-data digest, risk metrics,
quality results, and limitations. A deterministic 16-character fingerprint changes when a
material input, data series, or reported result changes. The generated timestamp is excluded
from that fingerprint, so recreating the same analysis later does not change its identity.
Full simulated path arrays are intentionally excluded; terminal scenarios remain available
as a separate CSV in the single-investment lab.

### Historical stress lab

After running a simulation, open **Historical stress** to compare its VaR cutoffs with
the ticker's actual rolling outcomes over the same holding period. Unlike either
bootstrap, this view preserves the complete original sequence of market days. It reports the
worst realized period, the average of the worst 5% of realized periods, threshold
crossings, and distinct severe episodes without repeatedly listing overlapping windows.

The chart uses every overlapping rolling window, so its frequencies are descriptive.
They are not independent observations and should not be interpreted as precise future
probabilities.

### Market regimes

Open **Market regimes** to place the ticker's recent behavior into one of four transparent
historical labels: calm advance, calm decline, turbulent advance, or turbulent decline.
The label combines a rolling 63-day price trend with rolling annualized volatility.
At each date, "turbulent" means volatility is above the expanding historical median
available by that date.

The dashboard compares overlapping forward outcomes following each label, including the
median return, loss frequency, VaR, and Expected Shortfall. These are descriptive
historical comparisons, not trading signals or out-of-sample forecasts. The volatility
boundary changes as new observations arrive, and regimes can change quickly.

### Historical model check

After running a simulation, open **Did the model work?** to test the model against
past outcomes. The backtest uses the simulation's selected resampling model, includes
only information available at each historical checkpoint, and evaluates non-overlapping
forecast periods. It reports:

- actual versus expected VaR violations;
- a Kupiec unconditional-coverage test;
- a Christoffersen independence test for violation clustering;
- a combined conditional-coverage test for rate and timing;
- a Wilson confidence interval for the observed violation rate;
- predicted loss cutoffs against realized losses; and
- cumulative violations against a 95% binomial reference range.

A result that does not reject the coverage or independence claims is not proof that
the model is correct. Short samples have limited statistical power, changing markets
can make historical relationships obsolete, and clustered violations can reveal that
the model adapts too slowly during stress.

### Model comparison and robustness

Open **Compare models** to test whether the risk estimate survives reasonable changes in
assumptions. The lab compares the independent-day baseline with several moving-block
lengths, applies the same walk-forward 95% VaR diagnostics to each model, and reruns the
selected model over several nested historical windows.

The purpose is to expose **model risk**, not select an automatic winner. A model with a
higher VaR is not necessarily more accurate, and a backtest p-value above 0.05 does not
prove that a model is correct. Finite simulations add sampling noise, long forecast
horizons leave few independent backtest periods, and different lookbacks can represent
different market conditions.

### Risk-budget position sizing

Open **Size a position** to translate one simulated downside measure into an educational
single-ticker limit. You provide the total investable portfolio value, the maximum share
of that portfolio that may be lost in the selected modeled scenario, and a separate
single-position concentration cap. RiskForge calculates:

```text
loss-budget limit = portfolio loss budget / modeled loss rate
final position limit = min(loss-budget limit, concentration limit)
```

Available yardsticks include 95% and 99% VaR, 95% and 99% Expected Shortfall, and the
95th-percentile maximum drawdown. The output is a scenario guardrail, not a recommended
trade or a claim that the position is safe. It does not optimize expected return and does
not account for correlations or overlapping exposures elsewhere in a portfolio. FINRA's
investor guidance explains why [risk tolerance is personal](https://www.finra.org/investors/insights/know-your-risk-tolerance)
and why [concentration can amplify losses](https://www.finra.org/investors/insights/concentration-risk).

### Volatility-aware filtered historical simulation

Open **Adaptive volatility** to compare the fixed historical bootstrap with an
EWMA-filtered historical simulation. The adaptive model estimates conditional volatility,
converts historical returns into standardized empirical shocks, resamples those shocks,
and scales them to the latest volatility state. With return innovation
\(\epsilon_t = r_t - \bar r\), RiskForge uses:

```text
sigma_t^2 = lambda * sigma_(t-1)^2 + (1 - lambda) * epsilon_(t-1)^2
z_t = epsilon_t / sigma_t
```

Along each simulated path it then applies:

```text
r_t* = historical mean + sigma_t* z_t*
sigma_(t+1)^2 = lambda * sigma_t^2 + (1 - lambda) * (r_t* - historical mean)^2
```

The default daily decay is 0.94, following the classic
[RiskMetrics Technical Document](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a).
The broader filtered-historical-simulation idea follows work by
[Barone-Adesi, Giannopoulos, and Vosper](https://doi.org/10.1111/1468-036X.00175).

The dashboard evaluates the fixed and adaptive models at identical historical checkpoints.
Each forecast uses only preceding returns, and realized evaluation periods do not overlap.
It reports violation rates plus coverage, independence, and combined p-values. A closer
historical violation rate is descriptive rather than proof of future superiority. Reusing
the same evaluation sample to choose the decay setting repeatedly would overfit the
backtest.

### Machine-learning volatility lab

Open **ML lab** to test whether a ridge-regression forecast adds value beyond the simpler
EWMA benchmark. The model uses daily closing-price features representing approximately
one week, one month, and one quarter of volatility history, along with recent downside
volatility, absolute return, trend, and the short-to-medium volatility ratio. These features
are inspired by the multi-horizon structure of the
[HAR volatility model](https://doi.org/10.1093/jjfinec/nbp001), but RiskForge uses daily
close-to-close returns rather than high-frequency realized-volatility data.

The evaluation is nested and chronological:

1. Forecast targets cover non-overlapping future periods.
2. The first 60% of examples form the initial training sample.
3. Ridge strength is chosen with expanding time-series splits inside that initial sample.
4. The final 40% remains untouched during tuning.
5. For each held-out prediction, the model is refit using only information available before
   that outcome.

This follows the time-aware validation principle demonstrated in scikit-learn's
[lagged-feature forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
RiskForge compares ML and EWMA mean absolute error and root mean squared error. It reports
when ML loses. Only after this evaluation does the current ML volatility forecast become the
starting volatility for a filtered Monte Carlo simulation; the empirical shocks and later
EWMA path updates remain unchanged.

The final current model may use all now-known labeled history, but its ridge strength remains
the one selected inside the original training period. Trying many feature sets against the
same held-out period will eventually overfit it, so results should be confirmed on a genuinely
new ticker or later time period.

### Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io).
3. Create an app from the repository and select `streamlit_app.py` as the entrypoint.
4. In advanced settings, select Python 3.12, then deploy.

The root `uv.lock` contains the complete deployment environment, so do not add a
second dependency file such as `requirements.txt`.

The repository includes `.github/workflows/ci.yml`. On every push and pull request it installs
the locked Python 3.12 environment, checks formatting and lint, runs the complete pytest and
Streamlit AppTest suite, and executes `riskforge health`. The workflow follows the official
[uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/).

## Model

For adjusted closing price \(P_t\), the continuously compounded daily return is

```text
r_t = log(P_t / P_(t-1)).
```

The default **moving-block bootstrap** samples overlapping blocks of consecutive returns
with replacement, joins the blocks, and trims them to the chosen horizon. This retains
short-run dependence inside each block. It follows the block-resampling idea introduced
by [Künsch (1989)](https://doi.org/10.1214/aos/1176347265). The block length is a modeling
choice, not a fitted truth.

The **IID bootstrap** samples each daily return independently with replacement. It is a
useful baseline but discards the order of market moves. A moving-block size of one is
equivalent to the IID sampler.

Given sampled returns \(r_1^*, ..., r_h^*\), a simulated price is

```text
P_h^* = P_0 exp(sum(r_t^*)).
```

The `seed` initializes NumPy's random generator, making otherwise identical runs
reproducible.

## Risk definitions

Terminal simple return and loss are

```text
R = P_h / P_0 - 1
L = -R.
```

- **Value at Risk (VaR):** empirical confidence quantile of terminal loss. Positive
  values denote losses; negative values denote gains at that quantile.
- **Expected Shortfall (ES):** mean simulated loss greater than or equal to VaR.
- **Probability of loss:** fraction of paths with a negative terminal return.
- **Threshold probability:** fraction whose terminal loss is at least the configured
  loss threshold.
- **Maximum drawdown:** largest peak-to-trough decline within each simulated path,
  reported as a positive fraction. RiskForge reports its mean and 95th percentile.

## Assumptions

- The selected historical window is representative of the future.
- Returns are sufficiently stable for historical resampling to be informative.
- Under the moving-block model, dependence beyond the selected block length is ignored.
- Under the IID model, daily log returns are treated as independent and identically
  distributed.
- Adjusted close incorporates splits and distributions as supplied by Yahoo Finance.
- There are no transaction costs, taxes, liquidity constraints, or execution effects.
- Single-ticker analysis represents one unhedged long position.
- Portfolio analysis is long-only, fully invested, buy-and-hold, and uses one common starting
  currency for hypothetical dollar allocations.
- Portfolio return rows are aligned on shared dates and sampled jointly; relationships absent
  from the selected history are not invented.
- Factor betas are constant linear estimates over one historical window, and entered shocks
  act on proxy prices while all other modeled relationships are held fixed.
- Allocation candidates are long-only, fully invested, evaluated on a finite shared scenario
  set, and assumed to move immediately from the current starting weights without trading cost.

## Limitations

Moving blocks preserve only short stretches of dependence and introduce artificial
boundaries between blocks. The block length and lookback window can materially change
results. The EWMA model reacts to volatility but still uses one fixed decay, assumes the
historical shock distribution remains informative, and does not predict structural breaks.
The ridge model is a small-data statistical learner, not an artificial-intelligence oracle;
it uses only price-derived features and can lose to EWMA or fail after a regime shift. The
IID baseline discards all serial dependence. Yahoo Finance is a convenient research source,
not an institutional market-data feed. Rare events absent from the historical sample cannot
be invented by bootstrap resampling. VaR also says nothing about the magnitude of losses
beyond its cutoff; Expected Shortfall is included for that reason.

The portfolio model does not convert listing currencies, model FX hedges, estimate expected
returns, or optimize allocations. The factor lab can expose historical relationships with
selected proxies, but it cannot prove that those proxies represent the portfolio's true
economic, sector, or macro exposures. Historical correlations and betas are unstable and can
shift during selloffs. Mixing tickers quoted in different currencies produces an incomplete
analysis unless currency risk is modeled separately. A larger ticker count is not proof of
economic diversification.

The allocation sandbox is not walk-forward validated. Optimized weights can be unstable and
can concentrate on errors in estimated means, covariance, or tail scenarios. Expected-return
shrinkage, weight caps, and turnover caps reduce some sensitivity but cannot eliminate model
risk. Scenario ES and drawdown limits apply only to the finite paths generated for that run;
they are not real-world loss guarantees.

RiskForge is for education and research, not investment advice.

## Development

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run riskforge health
```

The package uses a `src/` layout:

```text
src/riskforge/
├── allocation.py
├── backtesting.py
├── cli.py
├── data.py
├── factor_attribution.py
├── health.py
├── metrics.py
├── ml_volatility.py
├── outcomes.py
├── portfolio.py
├── position_sizing.py
├── quality.py
├── regimes.py
├── reporting.py
├── robustness.py
├── returns.py
├── simulation.py
├── stress.py
├── streamlit_components.py
├── visualization.py
└── volatility.py
```
