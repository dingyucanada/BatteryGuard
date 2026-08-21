"""Offline data adapters for BatteryGuard's canonical tabular contract."""

from batteryguard.ingestion.matr import MATRAdapter, load_matr
from batteryguard.ingestion.registry import AdapterRegistry, default_registry
from batteryguard.ingestion.splits import (
    SplitBuildError,
    build_cell_level_split,
    build_protocol_holdout_split,
    build_split_manifest,
)
from batteryguard.ingestion.standard import (
    CanonicalDataset,
    DataIngestionError,
    load_standardized,
    write_standardized,
)

__all__ = [
    "AdapterRegistry",
    "CanonicalDataset",
    "DataIngestionError",
    "MATRAdapter",
    "SplitBuildError",
    "build_cell_level_split",
    "build_protocol_holdout_split",
    "build_split_manifest",
    "default_registry",
    "load_matr",
    "load_standardized",
    "write_standardized",
]
