import pandas as pd
import pytest

from riskforge.data import MarketDataError, download_adjusted_price_frame


def _multi_asset_download_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=7)
    columns = pd.MultiIndex.from_product([["Close"], ["SPY", "TLT"]])
    return pd.DataFrame(
        [
            [100.0, 90.0],
            [101.0, 91.0],
            [102.0, 92.0],
            [103.0, 93.0],
            [104.0, 94.0],
            [105.0, 95.0],
            [106.0, 96.0],
        ],
        index=dates,
        columns=columns,
    )


def test_download_adjusted_price_frame_aligns_and_orders_tickers(monkeypatch) -> None:
    monkeypatch.setattr(
        "riskforge.data.yf.download",
        lambda *args, **kwargs: _multi_asset_download_frame(),
    )

    prices = download_adjusted_price_frame(["tlt", "spy"], lookback=4)

    assert list(prices.columns) == ["TLT", "SPY"]
    assert len(prices) == 5
    assert prices.iloc[-1].to_dict() == {"TLT": 96.0, "SPY": 106.0}


def test_download_adjusted_price_frame_rejects_duplicate_tickers() -> None:
    with pytest.raises(ValueError, match="unique"):
        download_adjusted_price_frame(["SPY", "spy"], lookback=4)


def test_download_adjusted_price_frame_requires_shared_history(monkeypatch) -> None:
    frame = _multi_asset_download_frame()
    frame.loc[frame.index[-4:], ("Close", "TLT")] = float("nan")
    monkeypatch.setattr("riskforge.data.yf.download", lambda *args, **kwargs: frame)

    with pytest.raises(MarketDataError, match="shared daily returns"):
        download_adjusted_price_frame(["SPY", "TLT"], lookback=4)
