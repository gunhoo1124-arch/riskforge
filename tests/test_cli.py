from typer.testing import CliRunner

from riskforge.cli import app

runner = CliRunner()


def test_simulate_is_an_explicit_command() -> None:
    result = runner.invoke(app, ["simulate", "--help"])

    assert result.exit_code == 0
    assert "riskforge simulate" in result.output
    assert "--horizon" in result.output
    assert "--model" in result.output
    assert "--block-size" in result.output
    assert "moving-block" in result.output


def test_health_command_reports_offline_readiness() -> None:
    result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "RiskForge health" in result.output
    assert "Overall status: PASS" in result.output
