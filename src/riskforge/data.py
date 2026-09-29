"""Market-data access and validation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    """Raised when usable market data cannot be downloaded."""


def _download_window(lookback: int) -> tuple[str, str]:
    """Return buffered calendar dates for a trading-day lookback."""
    if lookback < 2:
        raise ValueError("Lookback must be at least 2 trading days.")

    calendar_days = math.ceil(lookback * 365.25 / 252) + 30
    end = datetime.now(UTC).date() + timedelta(days=1)
    start = end - timedelta(days=calendar_days)
    return start.isoformat(), end.isoformat()


def _clean_tickers(tickers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate and standardize a multi-asset ticker collection."""
    symbols = tuple(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    if len(symbols) < 2:
        raise ValueError("A portfolio needs at least two tickers.")
    if len(symbols) > 10:
        raise ValueError("Use no more than 10 tickers in one portfolio analysis.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Each portfolio ticker must be unique.")
    return symbols


def download_adjusted_prices(ticker: str, lookback: int = 1_000) -> pd.Series:
    """Download recent adjusted daily closing prices for ``ticker``.

    Args:
        ticker: A symbol accepted by Yahoo Finance.
        lookback: Number of daily returns required. One additional price is returned
            so that exactly ``lookback`` returns can be calculated.

    Returns:
        A chronological series of positive adjusted closing prices.

    Raises:
        ValueError: If the inputs are invalid.
        MarketDataError: If Yahoo Finance returns insufficient or invalid data.
    """
    symbol = ticker.strip()
    if not symbol:
        raise ValueError("Ticker cannot be empty.")
    start, end = _download_window(lookback)

    try:
        frame = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # yfinance can surface several backend exceptions
        raise MarketDataError(f"Could not download data for {symbol}: {exc}") from exc

    if frame.empty or "Close" not in frame:
        raise MarketDataError(f"No adjusted closing prices were found for {symbol}.")

    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise MarketDataError(f"Received ambiguous closing-price data for {symbol}.")
        close = close.iloc[:, 0]

    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.name = symbol.upper()

    required_prices = lookback + 1
    if len(prices) < required_prices:
        available_returns = max(0, len(prices) - 1)
        raise MarketDataError(
            f"{symbol} has only {available_returns} usable daily returns; "
            f"{lookback} were requested. Try a smaller --lookback."
        )

    prices = prices.tail(required_prices).astype(float)
    if not (prices > 0).all():
        raise MarketDataError(f"Downloaded prices for {symbol} contain non-positive values.")
    return prices


def download_adjusted_price_frame(
    tickers: list[str] | tuple[str, ...],
    lookback: int = 1_000,
) -> pd.DataFrame:
    """Download aligned adjusted daily closing prices for multiple assets.

    Only dates with a valid price for every requested ticker are retained. This
    alignment is required so a resampled historical row represents one shared
    market day across the entire portfolio.

    Args:
        tickers: Two to ten unique symbols accepted by Yahoo Finance.
        lookback: Number of aligned daily returns required.

    Returns:
        A chronological frame with ``lookback + 1`` positive price rows and one
        column per requested ticker, in the requested order.

    Raises:
        ValueError: If tickers or lookback are invalid.
        MarketDataError: If complete aligned history cannot be downloaded.
    """
    symbols = _clean_tickers(tickers)
    start, end = _download_window(lookback)

    try:
        frame = yf.download(
            list(symbols),
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception as exc:  # yfinance can surface several backend exceptions
        joined = ", ".join(symbols)
        raise MarketDataError(f"Could not download portfolio data for {joined}: {exc}") from exc

    if frame.empty:
        raise MarketDataError("Yahoo Finance returned no portfolio price data.")

    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" in frame.columns.get_level_values(0):
            close = frame["Close"].copy()
        elif "Close" in frame.columns.get_level_values(1):
            close = frame.xs("Close", axis=1, level=1).copy()
        else:
            raise MarketDataError("Portfolio data did not contain adjusted closing prices.")
    elif "Close" in frame.columns and len(symbols) == 1:
        close = frame[["Close"]].copy()
        close.columns = [symbols[0]]
    else:
        raise MarketDataError("Portfolio data did not contain one price series per ticker.")

    close.columns = [str(column).upper() for column in close.columns]
    missing = [symbol for symbol in symbols if symbol not in close.columns]
    if missing:
        raise MarketDataError(
            "No adjusted closing prices were found for: " + ", ".join(missing) + "."
        )

    prices = close.loc[:, list(symbols)].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([float("inf"), float("-inf")], pd.NA)
    prices = prices.dropna(how="any").sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    required_prices = lookback + 1
    if len(prices) < required_prices:
        available_returns = max(0, len(prices) - 1)
        raise MarketDataError(
            f"Only {available_returns} shared daily returns were available across "
            f"{', '.join(symbols)}; {lookback} were requested. Remove a ticker with short "
            "history or choose a smaller history window."
        )

    prices = prices.tail(required_prices).astype(float)
    if not (prices > 0).all().all():
        raise MarketDataError("Downloaded portfolio prices contain non-positive values.")
    prices.index.name = "date"
    return prices
