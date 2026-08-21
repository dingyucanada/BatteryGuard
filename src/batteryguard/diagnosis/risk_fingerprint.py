"""Evidence-constrained degradation-risk fingerprinting.

The outputs describe statistical consistency, not microscopic mechanism proof.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from batteryguard.diagnosis.neighbors import NearestCellRetriever
from batteryguard.prediction.baselines import numeric_feature_frame
from batteryguard.schemas.prediction import RiskFingerprint


def _matching(columns: Iterable[str], include: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(token in column.lower() for token in include)]


def _positive_signal(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros(len(frame), dtype=float)
    values = frame.loc[:, columns].to_numpy(dtype=float)
    finite = np.where(np.isfinite(values), values, np.nan)
    with np.errstate(all="ignore"):
        result = np.nanmean(finite, axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class RiskFingerprinter:
    """Map interpretable early features to robust reference percentiles."""

    _COMPONENTS = (
        "polarization_risk",
        "thermal_stress",
        "efficiency_instability",
        "curve_shift",
    )

    def __init__(self, *, neighbors: int = 5) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.neighbors = int(neighbors)
        self.retriever = NearestCellRetriever()
        self.feature_names_in_: tuple[str, ...] = ()
        self.reference_signals_: dict[str, np.ndarray] = {}
        self.component_columns_: dict[str, tuple[str, ...]] = {}
        self.feature_medians_: dict[str, np.ndarray] = {}
        self.feature_scales_: dict[str, np.ndarray] = {}
        self.is_fitted_ = False

    def _fit_component_scaling(self, frame: pd.DataFrame) -> None:
        columns = [str(column) for column in frame.columns]
        selections = {
            "polarization_risk": _matching(
                columns,
                ("dcir", "charge_time_s_slope", "charge_time_s_change", "capacity_fade"),
            ),
            "thermal_stress": _matching(
                columns,
                ("max_temp_c_max", "max_temp_c_mean", "max_temp_c_slope", "avg_temp_c_max"),
            ),
            "efficiency_instability": _matching(
                columns,
                ("efficiency_instability_count", "efficiency_jump_count", "coulombic_efficiency_std"),
            ),
            "curve_shift": _matching(
                columns,
                ("delta_q_l1", "delta_q_l2", "delta_q_std", "delta_q_range"),
            ),
        }
        self.component_columns_ = {
            name: tuple(selected) for name, selected in selections.items()
        }
        for name, selected in selections.items():
            if not selected:
                self.feature_medians_[name] = np.asarray([], dtype=float)
                self.feature_scales_[name] = np.asarray([], dtype=float)
                continue
            matrix = frame.loc[:, selected].to_numpy(dtype=float)
            median = np.nanmedian(matrix, axis=0)
            q75 = np.nanpercentile(matrix, 75, axis=0)
            q25 = np.nanpercentile(matrix, 25, axis=0)
            scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
            self.feature_medians_[name] = np.nan_to_num(median, nan=0.0)
            self.feature_scales_[name] = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)

    def _signals(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        # Reuse train-derived median/IQR scaling for every query.
        standardized: dict[str, np.ndarray] = {}
        for name in self._COMPONENTS:
            selected = list(self.component_columns_[name])
            if not selected:
                standardized[name] = np.zeros(len(frame), dtype=float)
                continue
            matrix = frame.loc[:, selected].to_numpy(dtype=float)
            median = self.feature_medians_[name]
            scale = self.feature_scales_[name]
            z = (matrix - median) / scale
            standardized[name] = _positive_signal(
                pd.DataFrame(np.maximum(z, 0.0), columns=selected), selected
            )
        return standardized

    def fit(
        self,
        reference_features: pd.DataFrame | np.ndarray,
        cell_ids: Iterable[str] | None = None,
    ) -> RiskFingerprinter:
        frame, columns = numeric_feature_frame(reference_features)
        if len(frame) < 3:
            raise ValueError("risk reference requires at least three cells")
        self._fit_component_scaling(frame)
        self.reference_signals_ = self._signals(frame)
        self.retriever.fit(reference_features, cell_ids)
        self.feature_names_in_ = columns
        self.is_fitted_ = True
        return self

    def _percentile(self, component: str, value: float) -> float:
        reference = np.sort(self.reference_signals_[component])
        return float(np.searchsorted(reference, value, side="right") / (len(reference) + 1.0))

    def fingerprint(
        self,
        cell_id: str,
        features: pd.DataFrame | np.ndarray,
        *,
        ood_score: float,
        data_quality: float,
        evidence_ids: Iterable[str] = (),
    ) -> RiskFingerprint:
        if not self.is_fitted_:
            raise RuntimeError("fit RiskFingerprinter before fingerprint")
        if len(features) != 1:
            raise ValueError("fingerprint expects exactly one feature row")
        if not 0 <= ood_score <= 1 or not 0 <= data_quality <= 1:
            raise ValueError("ood_score and data_quality must be in [0, 1]")
        frame, _ = numeric_feature_frame(features, expected_columns=self.feature_names_in_)
        signals = self._signals(frame)
        scores = {
            component: self._percentile(component, float(signals[component][0]))
            for component in self._COMPONENTS
        }
        neighbor_rows = self.retriever.query(
            features,
            k=self.neighbors,
            exclude_cell_id=cell_id,
        )
        statements: list[str] = []
        if scores["polarization_risk"] >= 0.60:
            statements.append("Early observations are consistent with higher polarization / impedance risk.")
        if scores["thermal_stress"] >= 0.60:
            statements.append("The observed protocol is associated with higher thermal stress.")
        if scores["efficiency_instability"] >= 0.60:
            statements.append("Early coulombic efficiency is comparatively unstable.")
        if scores["curve_shift"] >= 0.60:
            statements.append("The early discharge curve shows a larger distribution-relative shift.")
        if ood_score >= 0.98:
            statements.append("Evidence is outside the training distribution; personalized claims should be withheld.")
        if not statements:
            statements.append("No elevated early-cycle risk dimension is supported relative to the reference set.")
        return RiskFingerprint(
            cell_id=str(cell_id),
            polarization_risk=scores["polarization_risk"],
            thermal_stress=scores["thermal_stress"],
            efficiency_instability=scores["efficiency_instability"],
            curve_shift=scores["curve_shift"],
            data_quality=float(data_quality),
            ood_score=float(ood_score),
            nearest_cells=[neighbor.cell_id for neighbor in neighbor_rows],
            statements=statements,
            evidence_ids=[str(value) for value in evidence_ids],
        )


def build_risk_fingerprint(
    reference_features: pd.DataFrame,
    query_features: pd.DataFrame,
    *,
    cell_id: str,
    ood_score: float,
    data_quality: float,
    reference_cell_ids: Iterable[str] | None = None,
    evidence_ids: Iterable[str] = (),
    neighbors: int = 5,
) -> RiskFingerprint:
    fingerprinter = RiskFingerprinter(neighbors=neighbors).fit(
        reference_features,
        reference_cell_ids,
    )
    return fingerprinter.fingerprint(
        cell_id,
        query_features,
        ood_score=ood_score,
        data_quality=data_quality,
        evidence_ids=evidence_ids,
    )
