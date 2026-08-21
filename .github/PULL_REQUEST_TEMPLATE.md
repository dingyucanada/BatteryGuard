## What changed

Describe the focused change and why it is needed.

## Authority and claim boundary

- [ ] No hardware actuation path was added.
- [ ] Models/optimizers still cannot create `ALLOW`.
- [ ] Synthetic results are labeled with sample size.
- [ ] No real/restricted data, hidden labels, credentials, ledgers, or local paths are included.

## Verification

- [ ] `uv run ruff check src tests apps`
- [ ] `uv run mypy src/batteryguard`
- [ ] `uv run pytest --cov=batteryguard --cov-report=term --cov-fail-under=82`
- [ ] Relevant negative/fault-injection tests added or updated

## Documentation

List affected contracts, safety claims, runbooks, or release notes.
