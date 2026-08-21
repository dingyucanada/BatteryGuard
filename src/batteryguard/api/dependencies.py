"""Dependency wiring for the local API and test overrides."""

from __future__ import annotations

from functools import lru_cache

from batteryguard.demo.engine import DemoEngine
from batteryguard.settings import AppSettings


@lru_cache(maxsize=1)
def get_engine() -> DemoEngine:
    return DemoEngine(settings=AppSettings.from_environment())


def reset_engine() -> None:
    """Drop the process singleton; intended for tests and explicit retraining."""

    get_engine.cache_clear()
