"""Offline health checks for local development and deployment smoke tests."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from riskforge import __version__


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """One deployment-readiness check."""

    name: str
    status: str
    detail: str


CRITICAL_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "plotly",
    "streamlit",
    "typer",
    "rich",
    "yfinance",
)


def run_health_checks(project_root: Path | None = None) -> tuple[HealthCheck, ...]:
    """Run deterministic checks that do not depend on a live market-data vendor."""
    checks: list[HealthCheck] = []
    python_ok = sys.version_info[:2] == (3, 12)
    checks.append(
        HealthCheck(
            "Python runtime",
            "pass" if python_ok else "fail",
            f"Running Python {sys.version_info.major}.{sys.version_info.minor}; 3.12 is required.",
        )
    )

    missing = [name for name in CRITICAL_MODULES if importlib.util.find_spec(name) is None]
    checks.append(
        HealthCheck(
            "Required packages",
            "fail" if missing else "pass",
            "Missing: " + ", ".join(missing) if missing else "All critical imports are available.",
        )
    )

    first = np.random.default_rng(42).normal(size=8)
    second = np.random.default_rng(42).normal(size=8)
    deterministic = bool(np.array_equal(first, second))
    checks.append(
        HealthCheck(
            "Deterministic random seed",
            "pass" if deterministic else "fail",
            "Repeated NumPy seeds reproduce the same sample.",
        )
    )
    checks.append(
        HealthCheck(
            "Package metadata",
            "pass" if __version__ else "fail",
            f"RiskForge version {__version__}.",
        )
    )

    if project_root is not None:
        root = project_root.resolve()
        entrypoint_exists = (root / "streamlit_app.py").is_file()
        lock_exists = (root / "uv.lock").is_file()
        checks.extend(
            [
                HealthCheck(
                    "Streamlit entrypoint",
                    "pass" if entrypoint_exists else "fail",
                    "streamlit_app.py is present."
                    if entrypoint_exists
                    else "streamlit_app.py was not found in the project root.",
                ),
                HealthCheck(
                    "Locked environment",
                    "pass" if lock_exists else "warning",
                    "uv.lock is present."
                    if lock_exists
                    else "uv.lock was not found; deployment may resolve different versions.",
                ),
            ]
        )
    return tuple(checks)


def health_status(checks: tuple[HealthCheck, ...]) -> str:
    """Return the most severe status across a health-check collection."""
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    return "pass"
