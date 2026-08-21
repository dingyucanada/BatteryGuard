# Contributing to BatteryGuard

Thanks for helping improve an audit-ready battery research prototype. BatteryGuard is intentionally simulation-only: contributions must not add a charger, cycler, BMS, vehicle, CAN, rack, storage-system, or other hardware actuation path.

## Development setup

```bash
uv sync --frozen --extra dev
uv run ruff check src tests apps
uv run mypy src/batteryguard
uv run pytest --cov=batteryguard --cov-report=term --cov-fail-under=82
uv run batteryguard demo --cell random --seed 42 --offline --no-reveal
```

## Pull-request expectations

1. Keep cell-level split boundaries and early-cycle leakage rules explicit.
2. Add negative tests for every new failure path, especially NaN, missing data, OOD, authentication, simulator failure, and safety violations.
3. Do not let a model, optimizer, or UI create an `ALLOW` result; only `SafetyShield` may do so.
4. Do not commit restricted datasets, hidden lifetime labels, credentials, model binaries, ledgers, generated reports, or local paths.
5. Label every synthetic metric as synthetic and include sample size. Never claim real-cell validity without a frozen external evaluation.
6. Update the relevant data contract, safety case, runbook, and test report when behavior changes.

## Commit and PR scope

Prefer small, reviewable changes. State the motivation, affected contracts, tests run, and any change in scientific or safety claims. A contribution that broadens the project's authority boundary requires an explicit design and safety review before implementation.

## Reporting a vulnerability

Do not open a public issue for a suspected security or blind-data disclosure problem. Follow [SECURITY.md](SECURITY.md).
