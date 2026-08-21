# BatteryGuard

[简体中文](README.zh-CN.md) · [Project site](https://dingyucanada.github.io/BatteryGuard/) · [Documentation](docs/PROJECT_SPEC.md) · [Demo runbook](docs/DEMO_RUNBOOK.md)

[![CI](https://github.com/dingyucanada/BatteryGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/dingyucanada/BatteryGuard/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-3776AB)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-0B6E4F)](LICENSE)
[![Research only](https://img.shields.io/badge/scope-simulation%20%2F%20research%20only-C23B22)](docs/SAFETY_CASE.md)

> **Simulation / Research Only.** BatteryGuard is an offline research prototype. Never connect it to a real charger, cycler, BMS, vehicle, battery rack, or storage system.

BatteryGuard uses the first 30 battery cycles to produce a calibrated lifetime estimate, an uncertainty interval, an out-of-distribution score, an explicit abstention decision, a degradation-risk fingerprint, and three simulated charging-policy families. A deterministic `SafetyShield` has final authority over every candidate. A controlled synthetic reveal workflow and append-only Evidence Ledger make both successes and failures auditable.

## What runs end to end

```text
early cycles → quality/leakage gate → model ladder → conformal interval
             → OOD/abstention → risk fingerprint → policy candidates
             → Twin-0 simulation → SafetyShield → Pareto set
             → controlled synthetic reveal → append-only evidence
```

The repository includes a deterministic demo dataset so, after dependencies are installed, the core loop runs without an LLM, external service, or proprietary data. The synthetic fixture demonstrates software behavior; it is not scientific validation. Its generator and outcomes are public, so the reveal is an API/access-control rehearsal, not an independent or secret blind benchmark. Real MATR/Severson data is intentionally not redistributed.

## Quick start

Requirements: Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
uv run batteryguard demo --cell random --seed 42 --offline --no-reveal
uv run streamlit run apps/streamlit_app.py
uv run uvicorn batteryguard.api.app:app --host 127.0.0.1 --port 8000
```

Container equivalents:

```bash
docker build -f docker/Dockerfile -t batteryguard:local .
docker run --rm -p 127.0.0.1:8000:8000 batteryguard:local
# Or start both the API and Streamlit UI:
export BATTERYGUARD_REVEAL_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose -f docker/compose.yaml up --build
```

The CLI never reveals by default. To exercise the synthetic access-control path, generate an explicit high-entropy evaluator token and opt in:

```bash
export BATTERYGUARD_REVEAL_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run batteryguard demo --cell random --seed 42 --offline --reveal
```

Without that environment variable, each process uses an undisclosed ephemeral token, so external reveal requests fail closed. Compose requires the variable and publishes ports on `127.0.0.1` only. These are local safeguards, not production authentication; the research API must not be exposed to an untrusted network.

## Verification snapshot

The 0.1.0 release-candidate verification snapshot includes unit, integration, regression, privacy, release-hardening, and safety tests, plus Ruff and strict Mypy checks. CI is configured to exercise Python 3.11–3.13 and the standard Dockerfile's non-root CLI/API/UI smoke path; its remote result remains a release gate until the first GitHub run completes. See [docs/TEST_REPORT.md](docs/TEST_REPORT.md) for the dated test count, coverage, commands, locked synthetic metrics, and limitations.

## Core guarantees

- Every split is grouped by `cell_id`; cross-split cells are a hard error.
- Features are limited to the configured early-cycle window and reject forbidden future-life fields.
- Every prediction contains a point estimate, conformal interval, OOD score, and abstention decision.
- OOD, low quality, wide intervals, missing signals, or simulator failure restrict or reject personalized aggressive policies.
- A learned model cannot set `ALLOW`; only the deterministic `SafetyShield` can.
- Runtime public-cell payloads omit `cycle_life`; a valid evaluator token is required for the reveal endpoint. Because the deterministic synthetic generator is public, this tests software access control rather than label secrecy.
- Evidence is append-only and SHA-256 chained with an atomic chain-head checkpoint; reveal updates append a new record instead of overwriting history.
- Repository configuration disables Streamlit and optional PyBaMM telemetry; after installation, the core demo is local and runtime-offline.

## Model ladder and evidence level

The training path compares B0 train median, B1 Ridge/Elastic Net, and B2 XGBoost on frozen group-aware splits. B2 remains only when locked evaluation supports it. Split conformal intervals are calibrated on a separate calibration set. Mahalanobis distance and rule metadata provide a transparent OOD baseline. Linux and Windows installs use the official CPU-only XGBoost distribution; macOS uses the standard distribution.

BatteryGuard reports observable **risk fingerprints**, not proven microscopic mechanisms. Wording such as “consistent with higher polarization risk” is allowed; claims that SEI growth, lithium plating, LAM, or LLI have been proven require dedicated measurements outside this MVP.

## Digital twin and safety

Twin-0 is a fast, deterministic empirical charge/thermal/degradation surrogate for offline demonstrations and fault injection. A PyBaMM adapter is optional (`uv sync --extra pybamm`) and never weakens the shield. The simulator is not a validated representation of a specific commercial cell.

The shield enforces voltage, C-rate, temperature, temperature rise/rate, SOC, finite complete trajectories, simulation convergence, plating-margin proxy, and OOD/abstention restrictions. Failed evaluation returns `REJECT` or `FALLBACK`, never optimistic success.

## Data

Place source data under `data/raw/` and follow [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md). Supported standardized inputs are CSV or Parquet `cells`, `cycles`, optional `timeseries`, `protocols`, and `splits` tables. The MATR adapter also recognizes the common Severson MATLAB batch structure; source layout mismatches fail with an actionable error. The HTTP ingest endpoint accepts only a relative directory contained under the configured `data/raw` root after symlink resolution; use the explicit local CLI workflow for other trusted paths.

The adapter is designed around the data structure described by Severson et al., [“Data-driven prediction of battery cycle life before capacity degradation”](https://doi.org/10.1038/s41560-019-0356-8), *Nature Energy* 4, 383–391 (2019). Data access, the original distribution's current terms, and any required citation remain the user's responsibility. Do not commit restricted or personal datasets.

## Project map

- `src/batteryguard/ingestion`, `quality`, `features`: data contract and leakage gates
- `prediction`, `uncertainty`, `ood`, `diagnosis`: model ladder and calibrated evidence
- `simulator`, `optimizer`, `safety`: policy simulation, Pareto selection, and final veto
- `evidence`, `demo`: audit ledger and controlled synthetic reveal
- `api`, `apps/streamlit_app.py`, `cli.py`: service, UI, and offline commands
- `tests/`: unit, integration, regression, and safety/fault-injection suites
- `docs/`: specification, safety case, demo runbook, and known limitations

## Known limitations

- The bundled dataset is synthetic and cannot establish battery-science accuracy.
- No MATR result is bundled or claimed. Any future evaluation on that dataset would remain limited to its LFP/graphite cells and experimental setting unless independently extended.
- Conformal coverage depends on calibration exchangeability and may not transfer across protocols or chemistries.
- Twin-0 metrics are transparent proxies, not certified electrochemical predictions.
- No hardware interface exists by design. Production use requires cell-specific calibration, independent validation, HIL, functional/expected-function safety, cybersecurity, software lifecycle controls, and applicable regulatory testing.
- Build release artifacts from a clean checkout (for example with `uv build`); never publish a raw developer working directory. Ignore files are not a substitute for scanning the release archive and full Git history for secrets, private data, models, ledgers, and local paths.

The FastAPI management and evidence routes are intended for a trusted local workstation and do not provide production-grade user authentication or tenancy isolation. See [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md), [docs/SAFETY_CASE.md](docs/SAFETY_CASE.md), and [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md) for acceptance details.
