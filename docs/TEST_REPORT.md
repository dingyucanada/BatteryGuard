# BatteryGuard Verification Report

Date: 2026-08-21
Release candidate: `0.1.0`
Scope: offline simulation/research prototype; no hardware control

## Result

BatteryGuard's complete offline path was implemented and exercised: grouped early-cycle data, leakage gates, B0/B1/B2 prediction, split-conformal uncertainty, Mahalanobis OOD and abstention, risk fingerprinting, deterministic policy simulation, independent safety veto, Pareto selection, a controlled synthetic reveal, and a hash-chained append-only evidence ledger.

All local source, package, and runtime smoke gates passed. The standard Dockerfile's cold online dependency fetch was environment-blocked on this host, as recorded below; the same Dockerfile and its non-root CLI/API/UI smoke path subsequently passed on the first green remote CI run. The numerical results use only the bundled deterministic synthetic fixture and are evidence of software behavior, not battery-science validity.

First green remote CI: [GitHub Actions run 32449835428](https://github.com/dingyucanada/BatteryGuard/actions/runs/32449835428), commit [`e2b64a8e0861771ce3c59e3e71bb62420169f3b6`](https://github.com/dingyucanada/BatteryGuard/commit/e2b64a8e0861771ce3c59e3e71bb62420169f3b6), completed 2026-08-21. Python 3.11, 3.12, and 3.13 all passed their locked-environment tests and offline no-reveal smoke; Python 3.13 also passed Ruff and strict Mypy. The dependent `container-smoke` job passed Compose validation, the standard Dockerfile build, non-root identity and CLI verification, FastAPI `/health`, and Streamlit `/_stcore/health`.

## Automated gates

| Gate | Command | Result |
| --- | --- | --- |
| Unit/integration/regression/privacy/safety | `uv run pytest -q` | PASS — 112 tests |
| Statement coverage | `uv run pytest --cov=batteryguard --cov-fail-under=82` | PASS — 85.95% |
| Remote Python matrix | GitHub Actions run `32449835428` | PASS — Python 3.11, 3.12, and 3.13 at commit `e2b64a8e0861771ce3c59e3e71bb62420169f3b6` |
| Lint/import rules | `uv run ruff check src tests apps` | PASS |
| Strict type checking | `uv run mypy src/batteryguard` | PASS — 70 source files |
| Locked environment | `uv sync --frozen --extra dev` | PASS |
| Lockfile consistency | `uv lock --check` | PASS |
| Installed dependency compatibility | `uv pip check` | PASS — 72 packages |
| Wheel and source package | `uv build` | PASS — Apache-2.0 `LICENSE` and `NOTICE` included |
| Release-hardening checks | `tests/safety/test_release_hardening.py` | PASS — license, token, container, workflow assertions |
| Data audit/train/calibrate/split/sim CLI | release commands | PASS — five exit codes 0 |
| Five-seed rehearsal | seeds 42, 43, 44, 45, 46 | PASS — five exit codes 0 |
| Controlled synthetic reveal | seed 42, cell `BG-0071` | PASS — explicit token, covered, ledger chain valid |
| FastAPI process and health | `GET /health` | PASS — HTTP 200 |
| Streamlit process and health | `GET /_stcore/health` | PASS — `ok` |
| Standard Dockerfile cold build (local Docker Desktop) | `docker build -f docker/Dockerfile ...` | ENVIRONMENT-BLOCKED — Docker Desktop's external package connection stalled; no source/lock error observed |
| Standard Dockerfile + non-root container smoke (remote CI) | GitHub Actions `container-smoke` | PASS — build, CLI, API, and Streamlit UI in run `32449835428` |
| Hash-verified Linux ARM64 runtime assembly | temporary multistage builder + `uv.lock` wheel hashes | PASS — image `sha256:28f1dcb3…095d`, Python 3.11.15, XGBoost CPU 3.2.0, no wheelhouse in final image |
| Non-root container identity | image config and `id -u` | PASS — `USER batteryguard`, UID 10001 |
| Hardened container API/CLI/UI smoke | no-reveal demo, `/health`, `/_stcore/health` | PASS — read-only root, all capabilities dropped, `no-new-privileges` |
| Compose validation | `docker compose -f docker/compose.yaml config -q` | PASS |

## Synthetic locked-test snapshot

The deterministic fixture contains 44 training cells, 10 conformal-calibration cells, 8 locked test cells, 6 controlled-reveal cells, and 4 external OOD cells. Cell IDs are disjoint across splits. The fixture generator and expected outcomes are public, so this is not an independent secret-label benchmark.

| Model | Test MAE (cycles) | Test RMSE (cycles) | Spearman |
| --- | ---: | ---: | ---: |
| B0 train median | 180.625 | 202.727 | 0.000 |
| B1 Ridge | 63.200 | 70.431 | 0.905 |
| B2 XGBoost | 33.691 | 40.228 | 0.952 |

For the selected B2 model, the nominal 90% split-conformal interval had 100% coverage on the eight synthetic locked-test cells, with mean width 298.505 cycles. The test set is deliberately small; this observed coverage is not a general coverage claim.

## Runtime acceptance evidence

- The offline CLI selected synthetic cell `BG-0071`, produced point estimate 865.707 cycles and interval `[716.455, 1014.960]`, and did not abstain. Reveal is off by default.
- At 25 °C, FAST, BALANCED, and LIFE were allowed by the deterministic shield.
- At 40 °C, FAST reached 45.406 °C and was rejected against the 45.0 °C limit; BALANCED and LIFE remained allowed and a conservative fallback was named.
- An explicit-token, one-time reveal returned the public synthetic lifetime of 885 cycles, absolute error 19.293 cycles, interval coverage `true`, status `SUPPORTED_IN_THIS_TEST`, and a new evidence record hash. This verifies runtime authorization and append semantics, not secret-label custody.
- Invalid or missing tokens are authenticated before cell/prediction lookup, so unauthenticated callers receive the same 401 response for known, unknown, and not-yet-predicted cell IDs.
- Invalid tokens, repeated reveal, cross-split cell leakage, forbidden future-life feature columns, non-finite/incomplete simulations, voltage/C-rate/temperature violations, OOD personalization, and ledger tampering are covered by negative tests.
- Generic serialization and pickle paths are tested to ensure `BlindPool` cannot export hidden lifetime labels.
- HTTP ingestion accepts only relative directories contained under `data/raw`; absolute paths, traversal, symlink escapes, and blank values are rejected without echoing a host path.
- The evidence JSONL hash chain is anchored by an atomically replaced `.head` checkpoint, which detects a clean deletion of complete tail records as well as ordinary in-chain mutation.
- Runtime telemetry is disabled by default for Streamlit and the optional PyBaMM backend. The static walkthrough's reveal button is a visual demonstration only: the 885-cycle outcome is embedded directly in the public HTML and no token is read.
- The local Docker Desktop external network repeatedly stalled during the ordinary `uv sync --frozen` build step. To avoid treating an old image as evidence, the smoke-tested Linux ARM64 image was assembled from freshly downloaded, lockfile-selected wheels with pip hash verification in a temporary multi-stage builder; the wheelhouse was discarded before the runtime stage. This validates the final source, dependency versions, non-root runtime, CLI, API, UI, and hardening flags, but not the standard Dockerfile's cold online fetch on this host. The ordinary Dockerfile build was independently validated by the first green remote CI run `32449835428` at commit `e2b64a8e0861771ce3c59e3e71bb62420169f3b6`.

## Boundaries and remaining scientific work

- No real MATR/Severson data is redistributed or claimed as validated here.
- Twin-0 outputs are deterministic engineering proxies, not certified electrochemical predictions.
- Risk fingerprints describe observable associations and do not prove SEI growth, lithium plating, LAM, or LLI.
- Conformal coverage depends on calibration exchangeability and must be revalidated after protocol, chemistry, site, or data-pipeline changes.
- There is intentionally no charger, cycler, BMS, vehicle, rack, or storage-system interface.
- The local ledger/checkpoint pair is tamper-evident, not a substitute for an external trust anchor. An attacker able to coherently rewrite both files requires controls such as signatures, remote witnessing, or WORM storage.
- Compose binds to loopback and applies container hardening, but the FastAPI management/evidence routes still lack production authentication, rate limits, quotas, and tenancy isolation. Do not expose this prototype to an untrusted network.
- Base/tool container images use version tags rather than immutable registry digests. A public container release still needs a reviewed SBOM, third-party notices, vulnerability/license/layer scans, and recorded image digest/provenance.
- Every public release must scan the complete Git history and verify branch/repository policy; a clean working-tree scan alone is insufficient. Publish only a clean-checkout artifact, never the raw developer working directory.
- Before any physical experiment: freeze a cell-specific safety case, complete independent data/model validation, HIL and fault injection, cybersecurity and lifecycle controls, and all applicable lab/regulatory review.
