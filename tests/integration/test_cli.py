from __future__ import annotations

import json

from typer.testing import CliRunner

from batteryguard.cli import app


def test_cli_offline_demo_defaults_to_no_reveal(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("BATTERYGUARD_REVEAL_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        ["demo", "--cell", "random", "--seed", "42", "--offline"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "offline"
    assert payload["research_only"] is True
    assert "reveal" not in payload
    assert "cycle_life" not in json.dumps(payload["cell"])
    assert payload["evidence_chain_valid"] is True


def test_cli_reveal_requires_explicit_environment_token(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("BATTERYGUARD_REVEAL_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["demo", "--reveal"])
    assert result.exit_code == 2
    assert "requires an explicit BATTERYGUARD_REVEAL_TOKEN" in result.output
