from __future__ import annotations

import json
import random

import pytest

from batteryguard.demo.engine import DemoEngine
from batteryguard.settings import AppSettings


@pytest.mark.integration
def test_offline_loop_and_five_seed_rehearsal(tmp_path) -> None:  # type: ignore[no-untyped-def]
    engine = DemoEngine(
        settings=AppSettings.from_environment(tmp_path),
        ledger_path=tmp_path / "evidence.jsonl",
    )
    ids = list(engine.blind_pool.cell_ids)
    for seed in range(42, 47):
        cell_id = random.Random(seed).choice(ids)
        assert "cycle_life" not in json.dumps(engine.public_cell(cell_id))
        prediction = engine.predict(cell_id)
        diagnosis = engine.diagnose(cell_id)
        policies = engine.policies(cell_id, ambient_temperature_c=40.0)
        assert prediction.evidence_ids
        assert diagnosis.evidence_ids
        assert policies.evidence_ids
        assert any(item.safety.decision.value != "ALLOW" for item in policies.policies)
        assert policies.pareto_front
    assert engine.ledger.verify_chain()


@pytest.mark.integration
def test_hidden_ood_cell_abstains_and_cannot_get_aggressive_personalization(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    engine = DemoEngine(
        settings=AppSettings.from_environment(tmp_path),
        ledger_path=tmp_path / "ood-evidence.jsonl",
    )
    ood_cell = next(
        cell["cell_id"]
        for cell in engine.list_blind_cells()
        if cell["chemistry"] != "LFP_graphite"
    )
    prediction = engine.predict(str(ood_cell))
    assert prediction.abstain
    assert "OUT_OF_DISTRIBUTION" in prediction.abstention_reasons
    policies = engine.policies(str(ood_cell), ambient_temperature_c=25.0)
    by_family = {item.policy.family.value: item for item in policies.policies}
    assert by_family["FAST"].safety.decision.value == "FALLBACK"
    assert by_family["BALANCED"].safety.decision.value == "FALLBACK"
    assert by_family["LIFE"].safety.decision.value == "ALLOW"
