from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from batteryguard.api.app import create_app
from batteryguard.api.dependencies import get_engine
from batteryguard.demo.engine import DemoEngine
from batteryguard.settings import AppSettings


def test_reveal_api_authorizes_before_cell_or_prediction_lookup(tmp_path: Path) -> None:
    settings = AppSettings.from_environment(tmp_path)
    engine = DemoEngine(settings=settings, ledger_path=tmp_path / "evidence.jsonl")
    application = create_app()
    application.dependency_overrides[get_engine] = lambda: engine
    known_cell = engine.blind_pool.cell_ids[0]
    unknown_cell = "not-in-the-blind-pool"

    with TestClient(application) as client:
        missing = client.post("/v1/demo/blind-reveal", json={"cell_id": known_cell})
        wrong_known = client.post(
            "/v1/demo/blind-reveal",
            json={"cell_id": known_cell, "evaluator_token": "wrong-known-secret"},
        )
        wrong_unknown = client.post(
            "/v1/demo/blind-reveal",
            json={"cell_id": unknown_cell, "evaluator_token": "wrong-unknown-secret"},
        )

        assert missing.status_code == wrong_known.status_code == wrong_unknown.status_code == 401
        assert missing.json() == wrong_known.json() == wrong_unknown.json() == {
            "detail": "valid evaluator token required"
        }
        assert "actual" not in (missing.text + wrong_known.text + wrong_unknown.text).lower()

        # Once authenticated, workflow-state and existence errors are safe to expose.
        before_prediction = client.post(
            "/v1/demo/blind-reveal",
            json={"cell_id": known_cell, "evaluator_token": settings.reveal_token},
        )
        authenticated_unknown = client.post(
            "/v1/demo/blind-reveal",
            json={"cell_id": unknown_cell, "evaluator_token": settings.reveal_token},
        )
        assert before_prediction.status_code == 422
        assert authenticated_unknown.status_code == 404

        prediction = client.post(f"/v1/cells/{known_cell}/predict", json={})
        assert prediction.status_code == 200
        claim_id = prediction.json()["evidence_ids"][0]

        revealed = client.post(
            "/v1/demo/blind-reveal",
            json={"cell_id": known_cell, "evaluator_token": settings.reveal_token},
        )
        duplicate = client.post(
            "/v1/demo/blind-reveal",
            json={"cell_id": known_cell, "evaluator_token": settings.reveal_token},
        )
        assert revealed.status_code == 200
        assert revealed.json()["actual_cycle_life"] > 0
        assert duplicate.status_code == 409
        assert "actual_cycle_life" not in duplicate.text

        claim_response = client.get(f"/v1/evidence/{claim_id}")
        assert claim_response.status_code == 200

    records = engine.ledger.records()
    rejected = [
        record for record in records if record.event_type == "BLIND_REVEAL_ATTEMPT_REJECTED"
    ]
    assert [record.payload["reason"] for record in rejected] == [
        "MISSING_TOKEN",
        "INVALID_TOKEN",
        "INVALID_TOKEN",
        "ALREADY_REVEALED",
    ]
    serialized_records = json.dumps(
        [record.model_dump(mode="json") for record in records]
    )
    assert "wrong-known-secret" not in serialized_records
    assert "wrong-unknown-secret" not in serialized_records
    assert engine.ledger.verify_chain()
