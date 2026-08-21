from __future__ import annotations

import numpy as np
import pandas as pd

from batteryguard.diagnosis import NearestCellRetriever, RiskFingerprinter


def _reference() -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, 12)
    return pd.DataFrame(
        {
            "cell_id": [f"R{i}" for i in range(12)],
            "dcir_ohm_mean": 0.02 + 0.002 * x,
            "charge_time_s_slope_per_cycle": 1 + x,
            "max_temp_c_max": 27 + 2 * x,
            "efficiency_instability_count": x * 2,
            "delta_q_l1_ah": 0.001 + 0.002 * x,
        }
    )


def test_neighbors_return_auditable_distance_and_exclude_self() -> None:
    reference = _reference()
    retriever = NearestCellRetriever().fit(reference)
    neighbors = retriever.query(reference.iloc[[0]], k=3, exclude_cell_id="R0")
    assert len(neighbors) == 3
    assert all(neighbor.cell_id != "R0" for neighbor in neighbors)
    assert neighbors[0].distance <= neighbors[1].distance


def test_high_query_produces_risk_fingerprint_without_mechanism_claims() -> None:
    reference = _reference()
    query = reference.iloc[[0]].copy()
    query["cell_id"] = "Q"
    for column in query.columns[1:]:
        query[column] = float(reference[column].max() * 3)
    fingerprint = RiskFingerprinter(neighbors=3).fit(reference).fingerprint(
        "Q",
        query,
        ood_score=0.2,
        data_quality=0.95,
        evidence_ids=["claim:risk:Q:v1"],
    )
    assert fingerprint.polarization_risk > 0.8
    assert fingerprint.thermal_stress > 0.8
    assert len(fingerprint.nearest_cells) == 3
    text = " ".join(fingerprint.statements).lower()
    assert "proven" not in text
    assert "confirmed" not in text
