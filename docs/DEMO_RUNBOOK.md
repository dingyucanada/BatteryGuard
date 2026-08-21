# Three-Minute Offline Demo Runbook

## Before the room

1. Run `uv sync --extra dev` once while online.
2. Run `uv run pytest` and save the summary.
3. Generate a fresh token, for example `export BATTERYGUARD_REVEAL_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"`.
4. Start Streamlit and switch the machine offline.
5. Rehearse at least five complete runs with seeds 42–46.
6. Tell the audience that the deterministic generator and the static 885-cycle walkthrough are public. The token exercises runtime access control only; it is not an independent blind evaluation.

## Story

**0:00–0:25 — runtime-redacted synthetic cell.** Select a demo cell. Confirm that the runtime public-cell JSON shows only its first 30 cycles and omits `cycle_life`. Do not describe the public fixture as secret.

**0:25–0:55 — prediction.** Show point estimate, 90% interval, OOD score, abstention, nearest cells, and risk wording. State that the bundled run is synthetic software evidence.

**0:55–1:35 — policy trade-offs.** Compare Fast, Balanced, and Life on charge time, temperature, degradation proxy, and energy loss. Show the Pareto set, not one universal optimum.

**1:35–2:05 — safety reversal.** Increase ambient temperature. Show that the apparently fastest policy is rejected and a conservative option remains. Open the precise violation and safety-case hash.

**2:05–2:40 — controlled synthetic reveal.** Enter the explicit evaluator token. Reveal the already-public synthetic outcome, interval coverage, and error. Explain that this demonstrates API authorization and one-time ledger append semantics; the Evidence Ledger adds a new record and never rewrites the prediction.

**2:40–3:00 — boundary.** Close with: “BatteryGuard is an audit-ready R&D co-pilot. It does not replace a certified BMS or physical validation.”

## Recovery

- If a live model path fails, use the deterministic cached fixture.
- If simulation fails, demonstrate the intended `FALLBACK` response.
- If the synthetic prediction misses, keep the failure visible; explain coverage and append-only evidence.
- For a future genuinely blind evaluation, keep evaluator-held labels outside this repository and publish only pre-approved aggregate results.
