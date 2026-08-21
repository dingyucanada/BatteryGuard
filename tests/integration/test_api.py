from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from batteryguard.api.app import create_app
from batteryguard.api.dependencies import get_engine
from batteryguard.demo.engine import DemoEngine
from batteryguard.settings import AppSettings


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> DemoEngine:
    root = tmp_path_factory.mktemp("api-engine")
    settings = AppSettings.from_environment(root)
    return DemoEngine(settings=settings, ledger_path=root / "evidence.jsonl")


@pytest.fixture(scope="module")
def client(engine: DemoEngine) -> TestClient:
    application = create_app()
    application.dependency_overrides[get_engine] = lambda: engine
    with TestClient(application) as test_client:
        yield test_client


def test_health_and_quality_contract(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["research_only"] is True
    quality = client.get("/v1/datasets/demo-synthetic-v1/quality")
    assert quality.status_code == 200
    assert quality.json()["leakage_checks_passed"] is True


def test_api_ingest_rejects_absolute_traversal_and_symlink_escape(
    client: TestClient, engine: DemoEngine
) -> None:
    raw_root = engine.settings.project_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    outside = engine.settings.project_root.parent / "outside-api-ingest"
    outside.mkdir(exist_ok=True)
    allowed = raw_root / "allowed"
    allowed.mkdir(exist_ok=True)
    assert engine._resolve_api_ingest_source("allowed") == allowed.resolve()
    escape_link = raw_root / "escape"
    try:
        escape_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this platform does not permit symlinks in the test environment")

    for source_path in (str(outside), "../outside-api-ingest", "escape", ""):
        response = client.post(
            "/v1/datasets/ingest",
            json={"source_path": source_path, "dataset_id": "rejected"},
        )
        assert response.status_code == 422
        assert str(engine.settings.project_root) not in response.text

    assert engine._last_ingested is None


def test_blind_payload_never_contains_lifetime(client: TestClient) -> None:
    response = client.get("/v1/cells/blind")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 6
    assert "cycle_life" not in json.dumps(payload)
    assert all(len(cell["early_cycles"]) == 30 for cell in payload)


def test_prediction_diagnosis_and_safety_reversal(
    client: TestClient, engine: DemoEngine
) -> None:
    cell_id = engine.blind_pool.cell_ids[0]
    prediction = client.post(f"/v1/cells/{cell_id}/predict", json={})
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["interval_low"] <= body["point_estimate"] <= body["interval_high"]
    assert 0 <= body["ood_score"] <= 1
    diagnosis = client.post(f"/v1/cells/{cell_id}/diagnose")
    assert diagnosis.status_code == 200
    assert diagnosis.json()["nearest_cells"]

    cool = client.post(
        f"/v1/cells/{cell_id}/policies",
        json={"ambient_temperature_c": 25.0, "initial_soc": 0.10},
    ).json()
    hot = client.post(
        f"/v1/cells/{cell_id}/policies",
        json={"ambient_temperature_c": 40.0, "initial_soc": 0.10},
    ).json()
    cool_decisions = {item["policy"]["family"]: item["safety"]["decision"] for item in cool["policies"]}
    hot_decisions = {item["policy"]["family"]: item["safety"]["decision"] for item in hot["policies"]}
    assert cool_decisions == {"FAST": "ALLOW", "BALANCED": "ALLOW", "LIFE": "ALLOW"}
    assert hot_decisions["FAST"] == "REJECT"
    assert hot_decisions["BALANCED"] == "ALLOW"
    assert hot_decisions["LIFE"] == "ALLOW"


def test_reveal_requires_token_and_appends_to_prediction_claim(
    client: TestClient, engine: DemoEngine
) -> None:
    cell_id = engine.blind_pool.cell_ids[1]
    prediction = client.post(f"/v1/cells/{cell_id}/predict", json={}).json()
    rejected = client.post(
        "/v1/demo/blind-reveal",
        json={"cell_id": cell_id, "evaluator_token": "wrong"},
    )
    assert rejected.status_code == 401
    assert "actual_cycle_life" not in rejected.text

    revealed = client.post(
        "/v1/demo/blind-reveal",
        json={"cell_id": cell_id, "evaluator_token": engine.settings.reveal_token},
    )
    assert revealed.status_code == 200
    assert revealed.json()["actual_cycle_life"] > 0
    claim = client.get(f"/v1/evidence/{prediction['evidence_ids'][0]}")
    assert claim.status_code == 200
    assert [record["event_type"] for record in claim.json()] == [
        "PREDICTION_CREATED",
        "BLIND_REVEAL_ATTEMPT_REJECTED",
        "BLIND_REVEAL_COMPLETED",
    ]
    assert engine.ledger.verify_chain()

    second = client.post(
        "/v1/demo/blind-reveal",
        json={"cell_id": cell_id, "evaluator_token": engine.settings.reveal_token},
    )
    assert second.status_code == 409


def test_unknown_blind_cell_is_404(client: TestClient) -> None:
    assert client.post("/v1/cells/not-a-cell/predict", json={}).status_code == 404
