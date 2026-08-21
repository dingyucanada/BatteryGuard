"""End-to-end offline BatteryGuard orchestration.

This module is the only place where data, prediction, uncertainty, OOD,
diagnosis, simulation, safety, evidence, and blind reveal are composed. It has
no hardware, network, LLM, or external-service dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import chi2

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from batteryguard.constants import (
    DATA_VERSION,
    RESEARCH_ONLY_NOTICE,
    SIMULATOR_VERSION,
    SOFTWARE_VERSION,
)
from batteryguard.demo.blind_pool import BlindCellNotFoundError, BlindPool
from batteryguard.demo.generate import DemoDataset, generate_demo_dataset
from batteryguard.demo.reveal import BlindRevealService, InvalidPredictionError
from batteryguard.diagnosis import RiskFingerprinter
from batteryguard.evidence import EvidenceLedger
from batteryguard.features import EarlyCycleFeaturePipeline
from batteryguard.ingestion import build_split_manifest, load_standardized
from batteryguard.ood import AbstentionPolicy, MahalanobisOOD
from batteryguard.optimizer import PolicySearch, SearchContext, default_policies
from batteryguard.prediction import ModelLadder, build_model, evaluate_regression
from batteryguard.quality import build_quality_report
from batteryguard.safety import SafetyShield
from batteryguard.schemas.data import DataQualityReport
from batteryguard.schemas.evidence import EvidenceStatus
from batteryguard.schemas.policy import (
    ChargingPolicy,
    EvaluatedPolicy,
    PolicyResponse,
)
from batteryguard.schemas.prediction import PredictionResponse, RiskFingerprint
from batteryguard.settings import AppSettings
from batteryguard.simulator import Twin0Simulator
from batteryguard.uncertainty import SplitConformalRegressor, empirical_coverage


class DemoEngine:
    """A deterministic CPU-only service suitable for CLI, API, UI, and tests."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        ledger_path: str | Path | None = None,
        model_name: str = "xgboost",
    ) -> None:
        self.settings = settings or AppSettings.from_environment()
        resolved_ledger = Path(ledger_path) if ledger_path is not None else (
            self.settings.project_root / "artifacts" / "evidence.jsonl"
        )
        self.ledger = EvidenceLedger(resolved_ledger)
        self.dataset_id = DATA_VERSION
        self._dataset: DemoDataset
        self._quality: DataQualityReport
        self._features: pd.DataFrame
        self._feature_columns: list[str]
        self._predictions: dict[str, PredictionResponse] = {}
        self._diagnoses: dict[str, RiskFingerprint] = {}
        self._policy_results: dict[tuple[str, float, float], PolicyResponse] = {}
        self._last_ingested: dict[str, Any] | None = None
        self._initialize(model_name=model_name, early_cycles=self.settings.prediction.early_cycles)

    def _initialize(self, *, model_name: str, early_cycles: int) -> None:
        self._dataset = generate_demo_dataset(
            seed=self.settings.seed,
            early_cycles=early_cycles,
        )
        self.dataset_id = self._dataset.dataset_id
        self._quality = build_quality_report(
            self.dataset_id,
            self._dataset.cells,
            self._dataset.cycles,
            self._dataset.timeseries,
            self._dataset.splits,
            early_cycles=early_cycles,
            hard_fail=True,
        )
        self.feature_pipeline = EarlyCycleFeaturePipeline(
            early_cycles=early_cycles,
            min_valid_cycles=min(10, early_cycles),
        )
        self._features = self.feature_pipeline.fit_transform(
            self._dataset.cells,
            self._dataset.cycles,
            self._dataset.timeseries,
        )
        self._feature_columns = [
            column for column in self._features.columns if column != "cell_id"
        ]
        self._fit_models(model_name)
        self.simulator = Twin0Simulator()
        self.shield = SafetyShield(self.settings.safety)
        self.policy_search = PolicySearch(self.simulator, self.shield)
        self._build_blind_services()
        self._predictions.clear()
        self._diagnoses.clear()
        self._policy_results.clear()

    def _rows_for_split(self, split: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        ids = self._dataset.splits.loc[
            self._dataset.splits["split"] == split, "cell_id"
        ].astype(str)
        frame = self._features[self._features["cell_id"].isin(ids)].copy()
        frame.sort_values("cell_id", inplace=True)
        labels = (
            self._dataset.cells.set_index("cell_id").loc[frame["cell_id"], "cycle_life"]
        )
        if labels.isna().any():
            raise ValueError(f"split {split!r} contains censored labels")
        return (
            frame[self._feature_columns],
            labels.to_numpy(dtype=float),
            frame["cell_id"].astype(str).tolist(),
        )

    def _fit_models(self, model_name: str) -> None:
        train_x, train_y, train_ids = self._rows_for_split("train")
        calibration_x, calibration_y, _ = self._rows_for_split("calibration")
        test_x, test_y, test_ids = self._rows_for_split("test")

        self.model_ladder = ModelLadder(
            model_kwargs={
                "B2": {
                    "n_estimators": 140,
                    "max_depth": 3,
                    "learning_rate": 0.045,
                    "random_state": self.settings.seed,
                }
            }
        ).fit(train_x, train_y)
        normalized = model_name.lower().replace("_", "-")
        if normalized in {"median", "b0", "train-median"}:
            self.model = self.model_ladder.models["B0"]
        elif normalized in {"ridge", "b1", "b1-ridge"}:
            self.model = self.model_ladder.models["B1"]
        elif normalized in {"xgboost", "xgb", "b2", "boosted-tree"}:
            self.model = self.model_ladder.models["B2"]
        else:
            self.model = build_model(model_name)
            self.model.fit(train_x, train_y)

        calibration_prediction = self.model.predict(calibration_x)
        self.conformal = SplitConformalRegressor(
            alpha=self.settings.prediction.alpha,
            clip_lower=1.0,
        ).calibrate(calibration_y, calibration_prediction)
        self.ood = MahalanobisOOD().fit(train_x)
        self.abstention = AbstentionPolicy(
            quality_threshold=self.settings.prediction.quality_threshold,
            ood_threshold=self.settings.prediction.ood_threshold,
            interval_width_ratio_max=self.settings.prediction.interval_width_ratio_max,
            required_early_cycles=self.feature_pipeline.early_cycles,
        )
        self.risk_fingerprinter = RiskFingerprinter(neighbors=5).fit(train_x, train_ids)

        split_lookup = self._dataset.cells.set_index("cell_id")
        test_protocols = split_lookup.loc[test_ids, "protocol_id"].astype(str).tolist()
        ladder_predictions = self.model_ladder.predict_all(test_x)
        self._ladder_report = {
            level: evaluate_regression(
                test_y,
                ladder_predictions[level],
                groups=test_protocols,
            ).to_dict()
            for level in ("B0", "B1", "B2")
        }
        main_prediction = np.asarray(self.model.predict(test_x), dtype=float)
        low, high = self.conformal.interval(main_prediction)
        coverage, width = empirical_coverage(test_y, low, high)
        self.model_version = (
            f"{self.model.model_name}:{self.feature_pipeline.feature_version}:seed{self.settings.seed}"
        )
        self._main_report = {
            "selected_model": self.model.model_name,
            "model_version": self.model_version,
            "feature_version": self.feature_pipeline.feature_version,
            "train_cells": len(train_x),
            "calibration_cells": len(calibration_x),
            "test_cells": len(test_x),
            "calibration_quantile": self.conformal.quantile_,
            "coverage_target": self.conformal.coverage_target,
            "test_coverage": coverage,
            "mean_interval_width": width,
            "synthetic_fixture_only": True,
        }

    def _build_blind_services(self) -> None:
        hidden_ids = set(
            self._dataset.splits.loc[
                self._dataset.splits["split"] == "demo_hidden", "cell_id"
            ].astype(str)
        )
        hidden_cells = cast(
            list[dict[str, Any]],
            self._dataset.cells[
                self._dataset.cells["cell_id"].isin(hidden_ids)
            ].to_dict(orient="records"),
        )
        early_by_cell = cast(
            dict[str, list[dict[str, Any]]],
            {
                cell_id: self._dataset.cycles[
                    self._dataset.cycles["cell_id"] == cell_id
                ].to_dict(orient="records")
                for cell_id in sorted(hidden_ids)
            },
        )
        self.blind_pool = BlindPool(hidden_cells, early_by_cell)
        self.reveal_service = BlindRevealService(
            self.blind_pool,
            self.ledger,
            self.settings.reveal_token,
            data_version=self.dataset_id,
            feature_version=self.feature_pipeline.feature_version,
            model_version=self.model_version,
            split_id="demo_hidden",
        )

    def _feature_row(self, cell_id: str) -> pd.DataFrame:
        self.blind_pool.get_public_cell(cell_id)
        row = self._features[self._features["cell_id"] == cell_id]
        if len(row) != 1:
            raise RuntimeError(f"expected one feature row for {cell_id}")
        return row[self._feature_columns]

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "mode": "offline-demo" if self.settings.offline else "research",
            "software_version": SOFTWARE_VERSION,
            "research_only": True,
            "evidence_chain_valid": self.ledger.verify_chain(),
        }

    def quality_report(self) -> dict[str, object]:
        return self._quality.model_dump(mode="json")

    def model_report(self) -> dict[str, object]:
        return {"main": self._main_report, "ladder": self._ladder_report}

    def list_blind_cells(self) -> list[dict[str, object]]:
        return self.blind_pool.list_public_cells()

    def public_cell(self, cell_id: str) -> dict[str, object]:
        return self.blind_pool.get_public_cell(cell_id)

    def predict(
        self,
        cell_id: str,
        *,
        quality_override: float | None = None,
    ) -> PredictionResponse:
        public = self.blind_pool.get_public_cell(cell_id)
        row = self._feature_row(cell_id)
        point = float(np.asarray(self.model.predict(row), dtype=float)[0])
        low_array, high_array = self.conformal.interval([point])
        interval_low = float(low_array[0])
        interval_high = float(high_array[0])
        distance = float(self.ood.distance(row)[0])
        ood_score = float(chi2.cdf(distance**2, df=max(1, len(self._feature_columns))))
        if public.get("chemistry") != "LFP_graphite":
            ood_score = max(ood_score, 0.995)
        quality = (
            float(quality_override)
            if quality_override is not None
            else float(self._quality.quality_score)
        )
        early_rows = public.get("early_cycles", [])
        observed_cycles = len(
            {int(row["cycle_index"]) for row in early_rows if "cycle_index" in row}
        )
        abstention = self.abstention.evaluate(
            quality_score=quality,
            ood_score=ood_score,
            point_estimate=point,
            interval_low=interval_low,
            interval_high=interval_high,
            observed_cycles=observed_cycles,
            protocol_available=bool(public.get("protocol_id")),
            model_version=self.model_version,
            evidence_model_version=self.model_version,
        )
        claim_id = f"claim:pred:{cell_id}:v1"
        self.ledger.append(
            claim_id=claim_id,
            event_type="PREDICTION_CREATED",
            claim=f"Cell {cell_id} predicted 80% EOL lifetime is {point:.1f} cycles",
            status=EvidenceStatus.PENDING,
            data_version=self.dataset_id,
            feature_version=self.feature_pipeline.feature_version,
            model_version=self.model_version,
            split_id="demo_hidden",
            payload={
                "cell_id": cell_id,
                "point_estimate": point,
                "prediction_interval": [interval_low, interval_high],
                "coverage_target": self.conformal.coverage_target,
                "ood_score": ood_score,
                "abstain": abstention.abstain,
                "abstention_reasons": list(abstention.reasons),
                "validation": "PENDING_LIFETIME_REVEAL",
            },
        )
        response = PredictionResponse(
            cell_id=cell_id,
            observed_cycles=observed_cycles,
            point_estimate=point,
            rul_estimate=max(0.0, point - observed_cycles),
            interval_low=interval_low,
            interval_high=interval_high,
            coverage_target=self.conformal.coverage_target,
            ood_score=ood_score,
            abstain=abstention.abstain,
            abstention_reasons=list(abstention.reasons),
            coverage_note=(
                "Split conformal calibration on held-out synthetic cells; coverage does not "
                "transfer automatically across protocols or chemistries."
            ),
            model_name=self.model.model_name,
            evidence_ids=[claim_id],
        )
        self._predictions[cell_id] = response
        return response

    def diagnose(self, cell_id: str) -> RiskFingerprint:
        prediction = self._predictions.get(cell_id) or self.predict(cell_id)
        row = self._feature_row(cell_id)
        claim_id = f"claim:risk:{cell_id}:v1"
        fingerprint = self.risk_fingerprinter.fingerprint(
            cell_id,
            row,
            ood_score=prediction.ood_score,
            data_quality=float(self._quality.quality_score),
            evidence_ids=[claim_id],
        )
        self.ledger.append(
            claim_id=claim_id,
            event_type="RISK_FINGERPRINT_CREATED",
            claim=f"Observable early-cycle risk fingerprint for cell {cell_id}",
            status=EvidenceStatus.PENDING,
            data_version=self.dataset_id,
            feature_version=self.feature_pipeline.feature_version,
            model_version=self.model_version,
            split_id="demo_hidden",
            payload=fingerprint.model_dump(mode="json"),
        )
        self._diagnoses[cell_id] = fingerprint
        return fingerprint

    def policies(
        self,
        cell_id: str,
        *,
        ambient_temperature_c: float = 25.0,
        initial_soc: float = 0.10,
    ) -> PolicyResponse:
        prediction = self._predictions.get(cell_id) or self.predict(cell_id)
        claim_id = f"claim:policy:{cell_id}:{ambient_temperature_c:.1f}:v1"
        candidates = [
            policy.model_copy(update={"personalized": True})
            for policy in default_policies(include_fallback=False)
        ]
        response = self.policy_search.response(
            cell_id,
            candidates,
            context=SearchContext(
                initial_soc=initial_soc,
                ambient_temperature_c=ambient_temperature_c,
                ood_score=prediction.ood_score,
                abstain=prediction.abstain,
            ),
            evidence_ids=[claim_id],
        )
        decisions = {
            evaluated.policy.family.value: evaluated.safety.decision.value
            for evaluated in response.policies
        }
        self.ledger.append(
            claim_id=claim_id,
            event_type="POLICY_SET_EVALUATED",
            claim=f"Simulated policy set for cell {cell_id} at {ambient_temperature_c:.1f} C",
            status=EvidenceStatus.PENDING,
            data_version=self.dataset_id,
            feature_version=self.feature_pipeline.feature_version,
            model_version=self.model_version,
            split_id="demo_hidden",
            simulation_version=SIMULATOR_VERSION,
            policy_version="fixed-mscc-library-v1",
            safety_result=",".join(f"{family}:{decision}" for family, decision in decisions.items()),
            payload={
                "cell_id": cell_id,
                "ambient_temperature_c": ambient_temperature_c,
                "initial_soc": initial_soc,
                "decisions": decisions,
                "pareto_front": response.pareto_front,
                "rejected": response.rejected,
                "fallback": response.fallback,
                "safety_case_hash": self.shield.safety_case_hash,
            },
        )
        self._policy_results[(cell_id, ambient_temperature_c, initial_soc)] = response
        return response

    def evaluate_policy(
        self,
        policy: ChargingPolicy,
        *,
        ambient_temperature_c: float,
        initial_soc: float,
        ood_score: float,
        abstain: bool,
    ) -> EvaluatedPolicy:
        return self.policy_search.evaluate_policy(
            policy,
            context=SearchContext(
                initial_soc=initial_soc,
                ambient_temperature_c=ambient_temperature_c,
                ood_score=ood_score,
                abstain=abstain,
            ),
        )

    def reveal(self, cell_id: str, evaluator_token: str | None) -> dict[str, object]:
        claim_id = f"claim:pred:{cell_id}:v1"
        self.reveal_service.authorize(
            cell_id,
            evaluator_token,
            claim_id=claim_id,
        )
        if not self.blind_pool.has_cell(cell_id):
            raise BlindCellNotFoundError(cell_id)
        prediction = self._predictions.get(cell_id)
        if prediction is None:
            raise InvalidPredictionError("run a blind prediction before reveal")
        return self.reveal_service.reveal(
            cell_id,
            evaluator_token,
            prediction.point_estimate,
            prediction.interval_low,
            prediction.interval_high,
            claim_id=prediction.evidence_ids[0],
        )

    def evidence(self, claim_id: str | None = None) -> list[dict[str, object]]:
        records = (
            self.ledger.records_for_claim(claim_id)
            if claim_id is not None
            else self.ledger.records()
        )
        return [record.model_dump(mode="json") for record in records]

    def _resolve_api_ingest_source(self, source_path: str) -> Path:
        """Resolve an API dataset path inside the configured ``data/raw`` root.

        The CLI has an explicit local-file workflow.  The HTTP surface is narrower:
        accepting an absolute or escaping path would turn it into a server-side file
        oracle.  Resolving before containment also rejects symlinks that point out of
        the allowed root.
        """

        if not source_path.strip():
            raise ValueError("source_path must be a non-empty relative directory under data/raw")
        requested = Path(source_path)
        if requested.is_absolute():
            raise ValueError("source_path must be relative to data/raw")
        project_root = self.settings.project_root.resolve()
        allowed_root = (project_root / "data" / "raw").resolve()
        try:
            allowed_root.relative_to(project_root)
            resolved = (allowed_root / requested).resolve()
            resolved.relative_to(allowed_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("source_path must resolve inside data/raw") from exc
        if not resolved.is_dir():
            raise ValueError("dataset directory was not found under data/raw")
        return resolved

    def ingest_dataset(self, source_path: str, dataset_id: str) -> dict[str, object]:
        resolved_source = self._resolve_api_ingest_source(source_path)
        dataset = load_standardized(resolved_source, dataset_id=dataset_id)
        manifest = dataset.splits
        if manifest is None:
            manifest = build_split_manifest(dataset.cells, strategy="cell", seed=self.settings.seed)
        report = build_quality_report(
            dataset.dataset_id,
            dataset.cells,
            dataset.cycles,
            dataset.timeseries,
            manifest,
            early_cycles=self.feature_pipeline.early_cycles,
            hard_fail=True,
        )
        details: dict[str, object] = {
            "dataset_id": dataset.dataset_id,
            "cells": len(dataset.cells),
            "cycles": len(dataset.cycles),
            "timeseries_rows": len(dataset.timeseries),
            "quality_score": report.quality_score,
            "leakage_checks_passed": report.leakage_checks_passed,
        }
        self._last_ingested = {"dataset": dataset, "manifest": manifest, "report": report}
        return details

    def build_splits(self, *, strategy: str, seed: int) -> dict[str, object]:
        normalized = strategy.lower().replace("_", "-")
        if normalized in {"cell-grouped", "cell-level"}:
            normalized = "cell"
        manifest = build_split_manifest(self._dataset.cells, strategy=normalized, seed=seed)
        counts = manifest["split"].value_counts().sort_index().to_dict()
        return {
            "strategy": normalized,
            "seed": seed,
            "cell_disjoint": manifest["cell_id"].is_unique,
            "counts": {str(key): int(value) for key, value in counts.items()},
        }

    def retrain(self, *, model_name: str, early_cycles: int) -> dict[str, object]:
        if early_cycles < 10 or early_cycles > 100:
            raise ValueError("early_cycles must lie in [10, 100]")
        self._initialize(model_name=model_name, early_cycles=early_cycles)
        return {
            "model_name": self.model.model_name,
            "model_version": self.model_version,
            "early_cycles": early_cycles,
            "feature_version": self.feature_pipeline.feature_version,
            "report": self._main_report,
        }

    @property
    def research_only_notice(self) -> str:
        return RESEARCH_ONLY_NOTICE
