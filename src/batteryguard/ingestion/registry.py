"""Small explicit ingestion adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from batteryguard.ingestion.matr import load_matr
from batteryguard.ingestion.standard import CanonicalDataset, DataIngestionError, load_standardized

Adapter = Callable[..., CanonicalDataset]


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}

    def register(self, name: str, adapter: Adapter) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("adapter name cannot be blank")
        if normalized in self._adapters:
            raise ValueError(f"adapter already registered: {normalized}")
        self._adapters[normalized] = adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def load(self, name: str, source: str | Path, **kwargs: Any) -> CanonicalDataset:
        normalized = name.strip().lower()
        try:
            adapter = self._adapters[normalized]
        except KeyError as exc:
            raise DataIngestionError(
                f"unknown adapter {name!r}; available={list(self.names())}"
            ) from exc
        return adapter(source, **kwargs)


default_registry = AdapterRegistry()
default_registry.register("standard", load_standardized)
default_registry.register("csv-parquet", load_standardized)
default_registry.register("matr", load_matr)
default_registry.register("severson", load_matr)
