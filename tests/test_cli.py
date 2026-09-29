from click import unstyle
from typer.testing import CliRunner

from riskforge.cli import app

runner = CliRunner()


def test_simulate_is_an_explicit_command() -> None:
    result = runner.invoke(app, ["simulate", "--help"])
    output = unstyle(result.output)

    assert result.exit_code == 0
    assert "riskforge simulate" in output
    assert "--horizon" in output
    assert "--model" in output
    assert "--block-size" in output
    assert "moving-block" in output


def test_health_command_reports_offline_readiness() -> None:
    result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "RiskForge health" in result.output
    assert "Overall status: PASS" in result.output
