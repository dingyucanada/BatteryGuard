"""Command-line entry point for offline research and reproducible demos."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, cast

import typer
import yaml

from batteryguard.demo.engine import DemoEngine
from batteryguard.demo.reveal import AlreadyRevealedError
from batteryguard.ingestion import (
    build_split_manifest,
    default_registry,
    write_standardized,
)
from batteryguard.quality import build_quality_report
from batteryguard.settings import AppSettings

app = typer.Typer(help="BatteryGuard — Simulation / Research Only")
data_app = typer.Typer(help="Data ingestion and quality gates")
split_app = typer.Typer(help="Cell/protocol grouped split construction")
sim_app = typer.Typer(help="Offline digital-twin commands")
app.add_typer(data_app, name="data")
app.add_typer(split_app, name="split")
app.add_typer(sim_app, name="sim")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise typer.BadParameter(f"config does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise typer.BadParameter("configuration root must be a mapping")
    return value


def _demo_engine() -> DemoEngine:
    return DemoEngine(settings=AppSettings.from_environment(Path.cwd()))


def _policy_summary(response: Any) -> dict[str, Any]:
    return {
        "pareto_front": list(response.pareto_front),
        "rejected": list(response.rejected),
        "fallback": response.fallback,
        "policies": [
            {
                "policy_id": item.policy.policy_id,
                "family": item.policy.family.value,
                "decision": item.safety.decision.value,
                "pareto_optimal": item.pareto_optimal,
                "metrics": (
                    item.trajectory.metrics.model_dump(mode="json")
                    if item.trajectory.metrics is not None
                    else None
                ),
                "violations": [
                    violation.model_dump(mode="json")
                    for violation in item.safety.violations
                ],
                "safety_case_hash": item.safety.safety_case_hash,
            }
            for item in response.policies
        ],
    }


@data_app.command("ingest")
def data_ingest(
    config: Path = typer.Option(Path("configs/data/matr.yaml"), exists=True),
    output: Path = typer.Option(Path("data/processed/matr-v1")),
) -> None:
    """Parse a local source, validate it, and write canonical Parquet tables."""

    values = _load_yaml(config)
    dataset_id = str(values.get("dataset_id", values.get("dataset", "matr-v1")))
    source = Path(str(values.get("source", "data/raw/matr")))
    adapter = str(values.get("adapter", "matr"))
    dataset = default_registry.load(adapter, source, dataset_id=dataset_id)
    if dataset.splits is None:
        dataset.splits = build_split_manifest(
            dataset.cells,
            strategy=str(values.get("split_strategy", "protocol-holdout")),
            seed=int(values.get("seed", 42)),
        )
    report = build_quality_report(
        dataset_id,
        dataset.cells,
        dataset.cycles,
        dataset.timeseries,
        dataset.splits,
        early_cycles=int(values.get("early_cycles", 30)),
        hard_fail=True,
    )
    written = write_standardized(dataset, output)
    typer.echo(
        _json(
            {
                "dataset_id": dataset_id,
                "quality": report.model_dump(mode="json"),
                "written": {name: str(path) for name, path in written.items()},
            }
        )
    )


@data_app.command("audit")
def data_audit(dataset: str = typer.Option("demo-synthetic-v1")) -> None:
    """Run the active offline dataset's quality and leakage report."""

    engine = _demo_engine()
    if dataset != engine.dataset_id:
        raise typer.BadParameter(
            "this command audits the active offline fixture; use 'data ingest' for local sources"
        )
    typer.echo(_json(engine.quality_report()))


@split_app.command("build")
def split_build(
    dataset: str = typer.Option("demo-synthetic-v1"),
    strategy: str = typer.Option("protocol-holdout"),
    seed: int = typer.Option(42),
) -> None:
    engine = _demo_engine()
    if dataset != engine.dataset_id:
        raise typer.BadParameter("dataset is not loaded in the offline engine")
    typer.echo(_json(engine.build_splits(strategy=strategy, seed=seed)))


@app.command("train")
def train(
    model: str = typer.Option("xgboost"),
    early_cycles: int = typer.Option(30, min=10, max=100),
) -> None:
    engine = _demo_engine()
    typer.echo(_json(engine.retrain(model_name=model, early_cycles=early_cycles)))


@app.command("calibrate")
def calibrate(
    method: str = typer.Option("split-conformal"),
    alpha: float = typer.Option(0.10, min=0.01, max=0.50),
) -> None:
    if method.lower().replace("_", "-") != "split-conformal":
        raise typer.BadParameter("only transparent split-conformal is supported")
    settings = AppSettings.from_environment(Path.cwd()).model_copy(
        update={
            "prediction": AppSettings.from_environment(Path.cwd()).prediction.model_copy(
                update={"alpha": alpha}
            )
        }
    )
    engine = DemoEngine(settings=settings)
    typer.echo(_json(engine.model_report()["main"]))


@sim_app.command("precompute")
def sim_precompute(
    config: Path = typer.Option(Path("configs/sim/twin0.yaml"), exists=True),
) -> None:
    """Evaluate the fixed policy grid for documented discrete temperatures."""

    _load_yaml(config)
    engine = _demo_engine()
    cell_id = engine.blind_pool.cell_ids[0]
    result = {
        str(temperature): engine.policies(
            cell_id, ambient_temperature_c=temperature
        ).model_dump(mode="json")
        for temperature in (15.0, 25.0, 35.0, 40.0)
    }
    target = Path("artifacts/simulation-cache.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json(result) + "\n", encoding="utf-8")
    typer.echo(_json({"status": "PRECOMPUTED", "path": str(target), "scenarios": 4}))


@app.command("demo")
def demo(
    cell: str = typer.Option("random", help="Synthetic demo cell ID or 'random'"),
    seed: int = typer.Option(42),
    offline: bool = typer.Option(True, "--offline/--online"),
    reveal: bool = typer.Option(False, "--reveal/--no-reveal"),
    ambient_temperature_c: float = typer.Option(40.0),
) -> None:
    """Run prediction → policy → safety reversal → optional authorized reveal."""

    if not offline:
        raise typer.BadParameter("the MVP demo is intentionally offline-only")
    configured_reveal_token = os.getenv("BATTERYGUARD_REVEAL_TOKEN") if reveal else None
    if reveal and not configured_reveal_token:
        raise typer.BadParameter(
            "--reveal requires an explicit BATTERYGUARD_REVEAL_TOKEN; "
            "generate a high-entropy evaluator secret first"
        )
    engine = _demo_engine()
    if cell == "random":
        cell_id = random.Random(seed).choice(list(engine.blind_pool.cell_ids))
    else:
        cell_id = cell
        if cell_id not in engine.blind_pool:
            raise typer.BadParameter(f"unknown demo cell: {cell_id}")
    prediction = engine.predict(cell_id)
    diagnosis = engine.diagnose(cell_id)
    baseline = engine.policies(cell_id, ambient_temperature_c=25.0)
    hot = engine.policies(cell_id, ambient_temperature_c=ambient_temperature_c)
    public_cell = engine.public_cell(cell_id)
    early_cycles = cast(list[dict[str, Any]], public_cell.pop("early_cycles", []))
    output: dict[str, Any] = {
        "mode": "offline",
        "research_only": True,
        "cell": {
            **public_cell,
            "observed_cycles": len(early_cycles),
            "early_discharge_capacity_ah": [
                row["discharge_capacity_ah"] for row in early_cycles
            ],
        },
        "prediction": prediction.model_dump(mode="json"),
        "risk_fingerprint": diagnosis.model_dump(mode="json"),
        "policies_25c": _policy_summary(baseline),
        "policies_hot": _policy_summary(hot),
        "evidence_chain_valid": engine.ledger.verify_chain(),
    }
    if reveal:
        try:
            output["reveal"] = engine.reveal(cell_id, configured_reveal_token)
        except AlreadyRevealedError as exc:
            output["reveal"] = {"status": "ALREADY_REVEALED", "detail": str(exc)}
    typer.echo(_json(output))


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000, min=1, max=65535),
) -> None:
    import uvicorn

    uvicorn.run("batteryguard.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
