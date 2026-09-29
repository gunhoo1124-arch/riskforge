from datetime import date

import numpy as np
import pandas as pd
import pytest

from riskforge.quality import (
    DataQualityError,
    QualityStatus,
    assess_price_quality,
    require_data_quality,
)


def test_valid_prices_pass_all_quality_checks() -> None:
    prices = pd.Series(
        [100.0, 101.0, 99.0, 102.0, 103.0],
        index=pd.bdate_range("2026-09-21", periods=5),
        name="TEST",
    )

    report = assess_price_quality(
        prices,
        minimum_returns=4,
        as_of=date(2026, 9, 28),
    )

    assert report.overall_status is QualityStatus.PASS
    assert report.observations == 5
    assert report.assets == 1
    assert not report.has_failures
    assert set(report.checks["status"]) == {"pass"}


def test_stale_data_and_extreme_move_are_non_blocking_warnings() -> None:
    prices = pd.Series(
        [100.0, 190.0, 188.0],
        index=pd.bdate_range("2025-01-02", periods=3),
        name="TEST",
    )

    report = assess_price_quality(
        prices,
        minimum_returns=2,
        as_of=date(2026, 9, 28),
        extreme_log_return=0.50,
    )

    assert report.overall_status is QualityStatus.WARNING
    assert report.has_warnings
    assert not report.has_failures
    require_data_quality(report)


def test_invalid_prices_fail_gate_with_clear_error() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, np.nan, 100.0], "B": [50.0, 50.0, 50.0]},
        index=pd.bdate_range("2026-09-21", periods=3),
    )

    report = assess_price_quality(prices, minimum_returns=4, as_of=date(2026, 9, 28))

    assert report.overall_status is QualityStatus.FAIL
    with pytest.raises(DataQualityError, match="failed validation"):
        require_data_quality(report)


def test_quality_parameters_are_validated() -> None:
    prices = pd.Series([100.0, 101.0], index=pd.bdate_range("2026-09-21", periods=2))

    with pytest.raises(ValueError, match="minimum_returns"):
        assess_price_quality(prices, minimum_returns=1)
    with pytest.raises(ValueError, match="stale_after_days"):
        assess_price_quality(prices, stale_after_days=-1)


def test_duplicate_asset_labels_fail_without_crashing() -> None:
    prices = pd.DataFrame(
        [[100.0, 80.0], [101.0, 81.0], [102.0, 82.0]],
        index=pd.bdate_range("2026-09-21", periods=3),
        columns=["DUP", "DUP"],
    )

    report = assess_price_quality(prices, minimum_returns=2, as_of=date(2026, 9, 28))

    assert report.overall_status is QualityStatus.FAIL
    label_check = report.checks.loc[report.checks["check"] == "Unique asset labels"].iloc[0]
    assert label_check["status"] == "fail"
