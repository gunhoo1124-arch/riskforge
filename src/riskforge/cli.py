"""Command-line interface for RiskForge."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from riskforge.data import MarketDataError, download_adjusted_prices
from riskforge.health import health_status, run_health_checks
from riskforge.metrics import RiskSummary, summarize_risk
from riskforge.quality import DataQualityReport, assess_price_quality, require_data_quality
from riskforge.returns import calculate_log_returns
from riskforge.simulation import BootstrapMethod, returns_to_price_paths, simulate_bootstrap
from riskforge.visualization import save_fan_chart

app = typer.Typer(
    name="riskforge",
    help="Simulate market outcomes and estimate risk using historical bootstrap sampling.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


def _safe_filename(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ticker.strip())
    return safe.strip("._") or "ticker"


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _summary_table(
    ticker: str,
    initial_price: float,
    horizon: int,
    paths: int,
    lookback: int,
    loss_threshold: float,
    method: BootstrapMethod,
    block_size: int,
    quality: DataQualityReport,
    summary: RiskSummary,
) -> Table:
    table = Table(title=f"RiskForge - {ticker.upper()}", header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Adjusted price", f"${initial_price:,.2f}")
    table.add_row("Historical observations", f"{lookback:,}")
    table.add_row("Horizon", f"{horizon:,} trading days")
    table.add_row("Simulated paths", f"{paths:,}")
    if method is BootstrapMethod.MOVING_BLOCK:
        effective_size = min(block_size, horizon)
        table.add_row("Sampling model", f"Moving blocks ({effective_size} days)")
    else:
        table.add_row("Sampling model", "Independent days (IID)")
    table.add_row("Data quality gate", quality.overall_status.value.title())
    table.add_section()
    table.add_row("95% VaR", _percent(summary.var_95))
    table.add_row("99% VaR", _percent(summary.var_99))
    table.add_row("95% Expected Shortfall", _percent(summary.expected_shortfall_95))
    table.add_row("99% Expected Shortfall", _percent(summary.expected_shortfall_99))
    table.add_row("Probability of loss", _percent(summary.probability_of_loss))
    table.add_row(
        f"Probability loss >= {_percent(loss_threshold)}",
        _percent(summary.probability_exceeding_threshold),
    )
    table.add_row("Mean maximum drawdown", _percent(summary.mean_maximum_drawdown))
    table.add_row("95th percentile maximum drawdown", _percent(summary.maximum_drawdown_95))
    return table


@app.callback()
def main() -> None:
    """Run transparent, reproducible Monte Carlo market-risk simulations."""


@app.command()
def simulate(
    ticker: Annotated[str, typer.Argument(help="Yahoo Finance ticker, such as SPY.")],
    horizon: Annotated[
        int, typer.Option("--horizon", min=1, help="Number of future trading days.")
    ] = 20,
    paths: Annotated[
        int, typer.Option("--paths", min=1, help="Number of Monte Carlo paths.")
    ] = 10_000,
    lookback: Annotated[
        int,
        typer.Option("--lookback", min=2, help="Historical daily returns to sample from."),
    ] = 1_000,
    seed: Annotated[
        int, typer.Option("--seed", min=0, help="NumPy random seed for reproducibility.")
    ] = 42,
    loss_threshold: Annotated[
        float,
        typer.Option(
            "--loss-threshold",
            min=0.0,
            max=1.0,
            help="Terminal loss threshold as a decimal fraction.",
        ),
    ] = 0.10,
    model: Annotated[
        BootstrapMethod,
        typer.Option(
            "--model",
            help="Resampling model: moving-block preserves short return sequences.",
        ),
    ] = BootstrapMethod.MOVING_BLOCK,
    block_size: Annotated[
        int,
        typer.Option(
            "--block-size",
            min=1,
            help="Consecutive days per sampled block for the moving-block model.",
        ),
    ] = 5,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination for the interactive HTML chart."),
    ] = None,
) -> None:
    """Run a historical-bootstrap simulation for TICKER."""
    try:
        prices = download_adjusted_prices(ticker, lookback)
        quality = assess_price_quality(prices, minimum_returns=lookback)
        require_data_quality(quality)
        log_returns = calculate_log_returns(prices)
        simulated_returns = simulate_bootstrap(
            log_returns.to_numpy(),
            horizon=horizon,
            paths=paths,
            method=model,
            block_size=block_size,
            seed=seed,
        )
        initial_price = float(prices.iloc[-1])
        price_paths = returns_to_price_paths(simulated_returns, initial_price)
        summary = summarize_risk(price_paths, loss_threshold)

        chart_path = output or Path(f"riskforge_{_safe_filename(ticker)}_fan_chart.html")
        saved_path = save_fan_chart(price_paths, ticker, chart_path)
    except (ValueError, TypeError, MarketDataError, OSError) as exc:
        error_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        _summary_table(
            ticker,
            initial_price,
            horizon,
            paths,
            len(log_returns),
            loss_threshold,
            model,
            block_size,
            quality,
            summary,
        )
    )
    console.print(f"\n[green]Fan chart saved to:[/green] {saved_path}")
    console.print("[dim]RiskForge is an educational model, not investment advice.[/dim]")


@app.command()
def health() -> None:
    """Run offline deployment and reproducibility checks."""
    checks = run_health_checks(Path.cwd())
    table = Table(title="RiskForge health", header_style="bold cyan")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")
    status_styles = {"pass": "green", "warning": "yellow", "fail": "red"}
    for check in checks:
        style = status_styles[check.status]
        table.add_row(check.name, f"[{style}]{check.status.upper()}[/{style}]", check.detail)
    console.print(table)
    overall = health_status(checks)
    console.print(f"Overall status: [{status_styles[overall]}]{overall.upper()}[/]")
    if overall == "fail":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
