"""Market-data quality checks used before risk calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum

import numpy as np
import pandas as pd


class QualityStatus(StrEnum):
    """Severity assigned to a data-quality check."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Auditable results from validating a price series or frame."""

    checks: pd.DataFrame
    overall_status: QualityStatus
    observations: int
    assets: int
    start_date: str | None
    end_date: str | None

    @property
    def has_failures(self) -> bool:
        """Return whether any blocking quality check failed."""
        return self.overall_status is QualityStatus.FAIL

    @property
    def has_warnings(self) -> bool:
        """Return whether at least one non-blocking warning was raised."""
        return bool((self.checks["status"] == QualityStatus.WARNING.value).any())


class DataQualityError(ValueError):
    """Raised when market data fails a blocking integrity check."""


def _as_of_date(as_of: date | datetime | pd.Timestamp | None) -> date:
    if as_of is None:
        return datetime.now(UTC).date()
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, pd.Timestamp):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    raise TypeError("as_of must be a date, datetime, pandas Timestamp, or None.")


def assess_price_quality(
    prices: pd.Series | pd.DataFrame,
    *,
    minimum_returns: int = 60,
    as_of: date | datetime | pd.Timestamp | None = None,
    stale_after_days: int = 7,
    extreme_log_return: float = 0.50,
) -> DataQualityReport:
    """Assess whether adjusted prices are fit for a historical risk simulation.

    Structural problems, missing or non-positive values, insufficient history, and
    constant price series are blocking failures. Stale observations and unusually
    large daily moves are warnings because they can be legitimate for some assets.
    """
    if minimum_returns < 2:
        raise ValueError("minimum_returns must be at least 2.")
    if stale_after_days < 0:
        raise ValueError("stale_after_days cannot be negative.")
    if not np.isfinite(extreme_log_return) or extreme_log_return <= 0:
        raise ValueError("extreme_log_return must be a positive finite number.")

    checks: list[dict[str, str]] = []

    def add(check: str, status: QualityStatus, detail: str) -> None:
        checks.append({"check": check, "status": status.value, "detail": detail})

    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("Prices must be a pandas Series or DataFrame.")

    frame = prices.to_frame() if isinstance(prices, pd.Series) else prices.copy()
    observations = len(frame)
    assets = frame.shape[1]
    start_date: str | None = None
    end_date: str | None = None

    if observations == 0 or assets == 0:
        add("Usable table", QualityStatus.FAIL, "The price table is empty.")
        checks_frame = pd.DataFrame(checks)
        return DataQualityReport(
            checks=checks_frame,
            overall_status=QualityStatus.FAIL,
            observations=observations,
            assets=assets,
            start_date=None,
            end_date=None,
        )
    add(
        "Usable table",
        QualityStatus.PASS,
        f"Found {observations:,} price rows for {assets:,} asset(s).",
    )

    labels_unique = not frame.columns.has_duplicates
    add(
        "Unique asset labels",
        QualityStatus.PASS if labels_unique else QualityStatus.FAIL,
        (
            "Every asset column has a unique label."
            if labels_unique
            else "Asset columns must have unique labels."
        ),
    )

    index_is_dates = isinstance(frame.index, pd.DatetimeIndex)
    index_dates_valid = index_is_dates and not frame.index.hasnans
    if index_dates_valid:
        start_date = frame.index.min().date().isoformat()
        end_date = frame.index.max().date().isoformat()
        ordered = frame.index.is_monotonic_increasing and not frame.index.has_duplicates
        add(
            "Dates ordered and unique",
            QualityStatus.PASS if ordered else QualityStatus.FAIL,
            (
                "Dates increase without duplicates."
                if ordered
                else "Price dates must increase and contain no duplicates."
            ),
        )
    else:
        add(
            "Date index",
            QualityStatus.FAIL,
            (
                "Price dates cannot contain missing timestamps."
                if index_is_dates
                else "Prices need a pandas DatetimeIndex for chronology and freshness checks."
            ),
        )

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite = bool(np.isfinite(values).all())
    add(
        "Complete finite prices",
        QualityStatus.PASS if finite else QualityStatus.FAIL,
        (
            "Every price is present and finite."
            if finite
            else "Missing, infinite, or non-numeric prices were found."
        ),
    )

    positive = finite and bool((values > 0).all())
    add(
        "Positive prices",
        QualityStatus.PASS if positive else QualityStatus.FAIL,
        "Every price is positive." if positive else "Prices must all be greater than zero.",
    )

    available_returns = max(0, observations - 1)
    enough_history = available_returns >= minimum_returns
    add(
        "History length",
        QualityStatus.PASS if enough_history else QualityStatus.FAIL,
        (f"Found {available_returns:,} returns; at least {minimum_returns:,} are required."),
    )

    changing_columns: list[str] = []
    if finite and observations >= 2:
        unique_counts = numeric.nunique(dropna=False).to_numpy()
        changing_columns = [
            str(column)
            for column, unique_count in zip(numeric.columns, unique_counts, strict=True)
            if unique_count > 1
        ]
    all_change = len(changing_columns) == assets
    constant_columns = [
        str(column) for column in numeric.columns if str(column) not in changing_columns
    ]
    add(
        "Price variation",
        QualityStatus.PASS if all_change else QualityStatus.FAIL,
        (
            "Every asset has changing historical prices."
            if all_change
            else "No price variation for: " + ", ".join(constant_columns) + "."
        ),
    )

    if index_dates_valid:
        last_date = frame.index.max().date()
        age_days = (_as_of_date(as_of) - last_date).days
        if age_days < 0:
            add(
                "Latest observation",
                QualityStatus.WARNING,
                f"The latest date is {abs(age_days)} day(s) after the assessment date.",
            )
        elif age_days > stale_after_days:
            add(
                "Latest observation",
                QualityStatus.WARNING,
                f"The latest price is {age_days} calendar days old; verify the ticker and feed.",
            )
        else:
            add(
                "Latest observation",
                QualityStatus.PASS,
                f"The latest price is {age_days} calendar day(s) old.",
            )

    if positive and observations >= 2:
        log_returns = np.log(numeric / numeric.shift(1)).iloc[1:]
        extreme_count = int((log_returns.abs() > extreme_log_return).sum().sum())
        add(
            "Extreme daily moves",
            QualityStatus.WARNING if extreme_count else QualityStatus.PASS,
            (
                f"Found {extreme_count:,} absolute daily log return(s) above "
                f"{extreme_log_return:.0%}; verify adjustments and corporate actions."
                if extreme_count
                else f"No absolute daily log returns exceeded {extreme_log_return:.0%}."
            ),
        )

    checks_frame = pd.DataFrame(checks, columns=["check", "status", "detail"])
    statuses = set(checks_frame["status"])
    overall = (
        QualityStatus.FAIL
        if QualityStatus.FAIL.value in statuses
        else QualityStatus.WARNING
        if QualityStatus.WARNING.value in statuses
        else QualityStatus.PASS
    )
    return DataQualityReport(
        checks=checks_frame,
        overall_status=overall,
        observations=observations,
        assets=assets,
        start_date=start_date,
        end_date=end_date,
    )


def require_data_quality(report: DataQualityReport) -> None:
    """Raise a clear error when a quality report contains blocking failures."""
    failed = report.checks.loc[report.checks["status"] == QualityStatus.FAIL.value, "detail"]
    if not failed.empty:
        raise DataQualityError("Market data failed validation: " + " ".join(failed.tolist()))
