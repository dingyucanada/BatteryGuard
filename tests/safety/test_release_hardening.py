from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).parents[2]


@pytest.mark.safety
def test_distribution_uses_complete_apache_license_and_notice() -> None:
    license_text = (_ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (_ROOT / "NOTICE").read_text(encoding="utf-8")
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert len(license_text.splitlines()) >= 190
    assert "Apache License" in license_text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in license_text
    assert all(f"   {section}." in license_text for section in range(1, 10))
    assert notice == "BatteryGuard\nCopyright 2026 Yu Ding\n"
    assert metadata["license"] == "Apache-2.0"
    assert metadata["license-files"] == ["LICENSE", "NOTICE"]


@pytest.mark.safety
def test_published_surfaces_have_no_shared_default_reveal_credential() -> None:
    old_default = "batteryguard-" + "evaluator-demo"
    public_paths = [
        _ROOT / "src",
        _ROOT / "apps",
        _ROOT / "configs",
        _ROOT / "docker",
        _ROOT / "docs",
        _ROOT / "site",
        _ROOT / "README.md",
        _ROOT / "README.zh-CN.md",
        _ROOT / "Makefile",
    ]
    matches: list[str] = []
    for path in public_paths:
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if (
                candidate.is_file()
                and candidate.suffix not in {".pyc"}
                and old_default in candidate.read_text(encoding="utf-8", errors="ignore")
            ):
                matches.append(str(candidate.relative_to(_ROOT)))
    assert matches == []


@pytest.mark.safety
def test_container_defaults_are_local_non_root_and_telemetry_off() -> None:
    compose_text = (_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    streamlit_config = tomllib.loads(
        (_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )

    assert "127.0.0.1:8000:8000" in compose_text
    assert "127.0.0.1:8501:8501" in compose_text
    assert "BATTERYGUARD_REVEAL_TOKEN:?" in compose_text
    assert "STREAMLIT_BROWSER_GATHER_USAGE_STATS" in compose_text
    assert "PYBAMM_DISABLE_TELEMETRY" in compose_text
    assert "USER batteryguard" in dockerfile
    assert streamlit_config["browser"]["gatherUsageStats"] is False


@pytest.mark.safety
def test_github_workflows_use_minimum_permissions_and_immutable_actions() -> None:
    ci_text = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pages_text = (_ROOT / ".github" / "workflows" / "pages.yml").read_text(
        encoding="utf-8"
    )
    ci = yaml.safe_load(ci_text)

    assert ci["permissions"] == {"contents": "read"}
    assert set(ci["jobs"]) == {"test", "container-smoke"}
    assert ci["jobs"]["test"]["timeout-minutes"] > 0
    assert ci["jobs"]["container-smoke"]["timeout-minutes"] > 0
    assert ci["jobs"]["test"]["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
    ]
    uses = re.findall(r"^\s*uses:\s*[^@\s]+@([^\s#]+)", ci_text + pages_text, re.MULTILINE)
    assert uses
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in uses)
