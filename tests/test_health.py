from riskforge.health import health_status, run_health_checks


def test_offline_health_checks_pass_in_project_environment() -> None:
    checks = run_health_checks()

    assert health_status(checks) == "pass"
    assert {check.name for check in checks} >= {
        "Python runtime",
        "Required packages",
        "Deterministic random seed",
        "Package metadata",
    }
