from __future__ import annotations

import json
import logging
from pathlib import Path

from batteryguard.logging import configure_logging, log_event
from batteryguard.settings import AppSettings


def test_settings_environment_overrides_and_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BATTERYGUARD_OFFLINE", "0")
    monkeypatch.setenv("BATTERYGUARD_SEED", "17")
    monkeypatch.setenv("BATTERYGUARD_REVEAL_TOKEN", "secret")
    settings = AppSettings.from_environment(tmp_path)
    assert settings.project_root == tmp_path
    assert not settings.offline
    assert settings.seed == 17
    assert settings.reveal_token == "secret"

    monkeypatch.delenv("BATTERYGUARD_OFFLINE")
    monkeypatch.delenv("BATTERYGUARD_SEED")
    monkeypatch.delenv("BATTERYGUARD_REVEAL_TOKEN")
    monkeypatch.chdir(tmp_path)
    defaults = AppSettings.from_environment()
    assert defaults.project_root == tmp_path
    assert defaults.offline
    assert defaults.seed == 42
    assert len(defaults.reveal_token) >= 32
    assert defaults.reveal_token != "-".join(("batteryguard", "evaluator", "demo"))
    assert defaults.reveal_token != AppSettings.from_environment().reveal_token


def test_structured_logging_configures_and_serializes(monkeypatch, caplog) -> None:
    calls: list[dict[str, object]] = []

    def fake_basic_config(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    configure_logging(logging.DEBUG)
    assert calls == [{"level": logging.DEBUG, "format": "%(message)s"}]

    logger = logging.getLogger("batteryguard-test-structured")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_event(logger, "fixture_ready", path=Path("artifact.json"), count=2)
    payload = json.loads(caplog.records[-1].message)
    assert payload == {"count": 2, "event": "fixture_ready", "path": "artifact.json"}
