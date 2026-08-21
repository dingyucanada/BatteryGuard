from __future__ import annotations

import ast
import socket
import subprocess
from pathlib import Path
from typing import NoReturn

import pytest

from batteryguard.optimizer.policy_space import balanced_policy
from batteryguard.safety.shield import SafetyShield
from batteryguard.schemas.policy import SafetyDecision
from batteryguard.simulator.surrogate import Twin0Simulator

_FORBIDDEN_RUNTIME_MODULES = {
    "bleak",
    "bluetooth",
    "can",
    "canopen",
    "httpx",
    "opcua",
    "pymodbus",
    "requests",
    "serial",
    "socket",
    "subprocess",
    "urllib",
}


@pytest.mark.safety
def test_safety_critical_packages_do_not_import_network_or_hardware_clients() -> None:
    package_root = Path(__file__).parents[2] / "src" / "batteryguard"
    violations: list[str] = []
    for package in ("simulator", "optimizer", "safety"):
        for source_path in sorted((package_root / package).glob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported = [node.module]
                for module_name in imported:
                    top_level = module_name.split(".", maxsplit=1)[0]
                    if top_level in _FORBIDDEN_RUNTIME_MODULES:
                        violations.append(f"{source_path.name}: {module_name}")

    assert violations == []


@pytest.mark.safety
def test_twin0_and_shield_complete_when_network_and_process_launch_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*_: object, **__: object) -> NoReturn:
        raise AssertionError("offline safety path attempted external I/O")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)

    policy = balanced_policy()
    trajectory = Twin0Simulator().simulate(policy)
    result = SafetyShield().evaluate(policy, trajectory)

    assert trajectory.status == "SUCCESS"
    assert result.decision == SafetyDecision.ALLOW
