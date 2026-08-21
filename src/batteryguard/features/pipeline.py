"""Composed, versioned, leakage-safe early-cycle feature pipeline."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from batteryguard.features.delta_q import compute_delta_q_features
from batteryguard.features.protocol import derive_protocol_features
from batteryguard.features.summary import summarize_early_cycles
from batteryguard.quality.leakage import assert_feature_frame_safe, assert_no_future_cycles


class FeatureExtractionError(ValueError):
    """Raised when early-cycle observations cannot produce a feature table."""


class EarlyCycleFeaturePipeline:
    """Extract one model row per cell without labels or identifiers-as-proxies."""

    def __init__(
        self,
        early_cycles: int = 30,
        *,
        reference_cycle: int = 2,
        start_cycle: int = 2,
        min_valid_cycles: int = 10,
        delta_q_grid_points: int = 128,
        strict_min_cycles: bool = False,
    ) -> None:
        if early_cycles < 2:
            raise ValueError("early_cycles must be at least 2")
        if not 1 <= start_cycle <= reference_cycle < early_cycles:
            raise ValueError("require start_cycle <= reference_cycle < early_cycles")
        if min_valid_cycles < 2 or min_valid_cycles > early_cycles:
            raise ValueError("min_valid_cycles must lie in [2, early_cycles]")
        self.early_cycles = int(early_cycles)
        self.reference_cycle = int(reference_cycle)
        self.start_cycle = int(start_cycle)
        self.min_valid_cycles = int(min_valid_cycles)
        self.delta_q_grid_points = int(delta_q_grid_points)
        self.strict_min_cycles = bool(strict_min_cycles)
        self.feature_version = f"early{early_cycles}-unfitted"
        self.feature_names_: tuple[str, ...] = ()

    def transform(
        self,
        cells: pd.DataFrame,
        cycles: pd.DataFrame,
        timeseries: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        required_cells = {"cell_id", "nominal_capacity_ah"}
        missing_cells = required_cells - set(cells.columns)
        if missing_cells:
            raise FeatureExtractionError(f"cells missing columns: {sorted(missing_cells)}")
        if cells["cell_id"].duplicated().any():
            raise FeatureExtractionError("cells must have one row per cell")
        if "cycle_index" not in cycles.columns:
            raise FeatureExtractionError("cycles is missing cycle_index")
        cycle_index = pd.to_numeric(cycles["cycle_index"], errors="coerce")
        if cycle_index.isna().any():
            raise FeatureExtractionError("cycles contains non-numeric cycle_index")
        bounded_cycles = cycles.loc[cycle_index <= self.early_cycles].copy()
        bounded_cycles["cycle_index"] = cycle_index.loc[bounded_cycles.index].astype(int)
        assert_no_future_cycles(bounded_cycles, self.early_cycles)
        summary = summarize_early_cycles(
            bounded_cycles,
            early_cycles=self.early_cycles,
            start_cycle=self.start_cycle,
        )
        base = cells[["cell_id", "nominal_capacity_ah"]].copy()
        base["cell_id"] = base["cell_id"].astype(str)
        frame = base.merge(summary, on="cell_id", how="left", validate="one_to_one")
        frame["observed_cycles"] = frame["observed_cycles"].fillna(0).astype(int)
        frame["early_cycle_completeness"] = np.clip(
            frame["observed_cycles"] / max(1, self.early_cycles - self.start_cycle + 1),
            0.0,
            1.0,
        )
        frame["minimum_cycles_met"] = (
            frame["observed_cycles"] >= self.min_valid_cycles
        ).astype(float)
        insufficient = frame.loc[frame["observed_cycles"] < self.min_valid_cycles, "cell_id"].tolist()
        if insufficient and self.strict_min_cycles:
            raise FeatureExtractionError(
                f"cells have fewer than {self.min_valid_cycles} valid early cycles: {insufficient[:10]}"
            )

        nominal = pd.to_numeric(frame["nominal_capacity_ah"], errors="coerce")
        if nominal.isna().any() or (nominal <= 0).any():
            raise FeatureExtractionError("nominal_capacity_ah must be finite and positive")
        capacity_columns = [
            column
            for column in frame.columns
            if (column.startswith("charge_capacity_ah_") or column.startswith("discharge_capacity_ah_"))
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        for column in capacity_columns:
            frame[f"normalized_{column}"] = pd.to_numeric(frame[column], errors="coerce") / nominal

        if timeseries is not None and not timeseries.empty:
            if "cycle_index" not in timeseries.columns:
                raise FeatureExtractionError("timeseries is missing cycle_index")
            ts_cycle = pd.to_numeric(timeseries["cycle_index"], errors="coerce")
            if ts_cycle.isna().any():
                raise FeatureExtractionError("timeseries contains non-numeric cycle_index")
            bounded_ts = timeseries.loc[ts_cycle <= self.early_cycles].copy()
            bounded_ts["cycle_index"] = ts_cycle.loc[bounded_ts.index].astype(int)
            assert_no_future_cycles(bounded_ts, self.early_cycles)
            delta = compute_delta_q_features(
                bounded_ts,
                early_cycles=self.early_cycles,
                reference_cycle=self.reference_cycle,
                comparison_cycle=self.early_cycles,
                grid_points=self.delta_q_grid_points,
                strict=False,
            )
            protocol = derive_protocol_features(
                bounded_ts,
                cells,
                early_cycles=self.early_cycles,
            )
            if not delta.empty:
                frame = frame.merge(delta, on="cell_id", how="left", validate="one_to_one")
            if not protocol.empty:
                frame = frame.merge(protocol, on="cell_id", how="left", validate="one_to_one")

        frame.drop(columns=["nominal_capacity_ah"], inplace=True)
        frame.sort_values("cell_id", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        assert_feature_frame_safe(frame, early_cycles=self.early_cycles)
        self.feature_names_ = tuple(column for column in frame.columns if column != "cell_id")
        payload = {
            "early_cycles": self.early_cycles,
            "reference_cycle": self.reference_cycle,
            "start_cycle": self.start_cycle,
            "min_valid_cycles": self.min_valid_cycles,
            "delta_q_grid_points": self.delta_q_grid_points,
            "columns": self.feature_names_,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        self.feature_version = f"early{self.early_cycles}-dq-{digest}"
        return frame

    def fit_transform(
        self,
        cells: pd.DataFrame,
        cycles: pd.DataFrame,
        timeseries: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.transform(cells, cycles, timeseries)
