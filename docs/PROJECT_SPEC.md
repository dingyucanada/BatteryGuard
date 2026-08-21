# BatteryGuard MVP Project Specification

## Mission

BatteryGuard turns the first 30 valid cycles into an auditable research decision: a calibrated lifetime range, OOD/abstention status, observable risk fingerprint, simulated charging-policy trade-offs, a deterministic safety decision, and a controlled synthetic outcome reveal.

The software is **Simulation / Research Only**. The repository intentionally contains no hardware-control path.

## Required end-to-end behavior

1. Validate canonical cell/cycle/timeseries tables and enforce a single split per `cell_id`.
2. Derive features only from `cycle_index <= early_cycles`; reject future-life fields.
3. Compare B0 median, B1 Ridge/Elastic Net, and B2 XGBoost on frozen group-aware splits.
4. Fit split conformal residuals on calibration cells only.
5. Return a point estimate, interval, target coverage, OOD score, and explicit abstention decision.
6. Produce a non-causal risk fingerprint and nearest reference cells.
7. Simulate Fast, Balanced, Life, and conservative fallback strategies locally.
8. Apply `SafetyShield` after simulation and before Pareto selection.
9. Write every material claim to a hash-chained append-only ledger.
10. Omit `cycle_life` from runtime public-cell payloads and require a valid evaluator token for the reveal endpoint. The public deterministic fixture is an access-control rehearsal, not a secret benchmark.
11. After dependencies are installed, run the core demo without runtime network access, an LLM, GPU, or external dataset; disable optional UI/simulator telemetry.

## Acceptance matrix

| Area | Acceptance condition | Automated evidence |
|---|---|---|
| Data | Duplicate cell/cycle keys, impossible ranges, or cross-split cells fail explicitly | quality/leakage tests |
| Leakage | `cycle_life`, future capacity/temperature, filenames, and cycles after the window cannot become features | regression tests |
| Prediction | Response always contains interval, OOD, abstention, and evidence ID | schema/integration tests |
| Calibration | Finite-sample conformal quantile uses a separate calibration set | uncertainty tests |
| Safety | All known violating policies are rejected; simulator/NaN/missing failures fall back | safety fault injection |
| Authority | Optimizer or learned model cannot directly create an `ALLOW` result | safety API tests |
| Controlled reveal | Runtime public payloads contain no `cycle_life`; wrong token fails; success is recorded once; docs do not claim independent label secrecy | reveal/privacy integration tests |
| Evidence | Hash chain detects edits and reveal appends instead of overwriting | ledger tests |
| Offline | `batteryguard demo --offline` completes locally | end-to-end regression |
| Product | API health and principal workflow endpoints respond under TestClient | API integration tests |
| API file boundary | HTTP ingest accepts only relative directories that resolve inside configured `data/raw`; absolute, traversal, and symlink escapes fail | API negative tests |

## Scientific claims

The synthetic fixture validates software paths only. Its generator and labels are public, including the static walkthrough result, so it cannot establish blind-test independence. Any scientific evaluation must use a licensed source dataset and a frozen split manifest controlled outside the public repository. Report MAE, RMSE, MAPE/relative error, Spearman, interval coverage and width, risk–coverage, ID/OOD gap, and protocol/lifetime strata. Retain B2 only when it improves locked grouped tests.

## Explicit non-goals

- real charger, cycler, BMS, vehicle, rack, or storage control;
- online reinforcement learning or real-time DFN optimization;
- universal chemistry transfer;
- confirmed microscopic degradation mechanisms from public cycling curves;
- production certification or compliance claims.
