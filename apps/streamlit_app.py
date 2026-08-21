"""BatteryGuard's local controlled-synthetic-reveal Streamlit demonstration."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from batteryguard.api.dependencies import get_engine
from batteryguard.constants import RESEARCH_ONLY_NOTICE
from batteryguard.demo.reveal import AlreadyRevealedError, RevealAuthorizationError
from batteryguard.schemas.policy import SafetyDecision

st.set_page_config(page_title="BatteryGuard", page_icon="🔋", layout="wide")
engine = get_engine()

st.markdown(
    """
    <style>
      .bg-banner {padding: 0.9rem 1.1rem; border-radius: 0.7rem;
                  background: #17212b; color: #e8f5f0; border-left: 7px solid #d99221;}
      .small-note {color: #5d6b78; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<div class="bg-banner"><b>{RESEARCH_ONLY_NOTICE}</b></div>', unsafe_allow_html=True)
st.title("BatteryGuard")
st.caption("Early-Life Degradation Intelligence and Safe Charging Co-Pilot")

public_cells = engine.list_blind_cells()
cell_ids = [str(cell["cell_id"]) for cell in public_cells]
with st.sidebar:
    st.header("Synthetic demo controls")
    selected_cell = st.selectbox("Demo cell", cell_ids, index=0)
    ambient_temperature = st.select_slider(
        "Ambient temperature (°C)", options=[15.0, 25.0, 35.0, 40.0, 42.0]
    )
    st.caption("42 °C is the planned safety-reversal scenario.")
    run_prediction = st.button("Run auditable prediction", type="primary", use_container_width=True)

if run_prediction or st.session_state.get("cell_id") != selected_cell:
    st.session_state["cell_id"] = selected_cell
    st.session_state["prediction"] = engine.predict(selected_cell)
    st.session_state["diagnosis"] = engine.diagnose(selected_cell)
    st.session_state.pop("policies", None)
    st.session_state.pop("reveal", None)

prediction = st.session_state.get("prediction")
diagnosis = st.session_state.get("diagnosis")
cell = engine.public_cell(selected_cell)

home, blind, prediction_tab, strategy, safety, reveal, evidence = st.tabs(
    ["Home", "Public Cell", "Prediction", "Strategy Lab", "Safety", "Reveal", "Evidence"]
)

with home:
    st.subheader("From a point estimate to a falsifiable decision loop")
    st.write(
        "BatteryGuard combines grouped data discipline, calibrated uncertainty, OOD/abstention, "
        "a deterministic charging surrogate, an independent safety veto, and controlled outcome reveal."
    )
    left, middle, right = st.columns(3)
    left.metric("Observed window", "30 cycles")
    middle.metric("Candidate families", "Fast · Balanced · Life")
    right.metric("Final authority", "SafetyShield")
    st.info(
        "The bundled dataset is synthetic and proves software behavior only. Scientific claims "
        "require licensed source data and frozen grouped evaluation."
    )
    report = engine.model_report()
    st.json(report, expanded=False)

with blind:
    st.subheader(f"Runtime-redacted synthetic cell · {selected_cell}")
    safe_metadata = {key: value for key, value in cell.items() if key != "early_cycles"}
    st.json(safe_metadata, expanded=False)
    early_frame = pd.DataFrame(cell["early_cycles"])
    st.line_chart(
        early_frame.set_index("cycle_index")[["discharge_capacity_ah"]],
        x_label="Cycle",
        y_label="Discharge capacity (Ah)",
    )
    st.caption(
        "The runtime public-cell payload omits the full-life target before authorization. "
        "The deterministic generator is public, so this is an access-control exercise, "
        "not a secret or independent blind evaluation."
    )

with prediction_tab:
    if prediction is None or diagnosis is None:
        st.warning("Run prediction from the sidebar.")
    else:
        st.subheader("Calibrated lifetime evidence")
        point, interval, ood, answer = st.columns(4)
        point.metric("Point estimate", f"{prediction.point_estimate:.0f} cycles")
        interval.metric(
            f"{prediction.coverage_target:.0%} interval",
            f"{prediction.interval_low:.0f}–{prediction.interval_high:.0f}",
        )
        ood.metric("OOD score", f"{prediction.ood_score:.3f}")
        answer.metric("Decision", "ABSTAIN" if prediction.abstain else "ANSWER")
        if prediction.abstain:
            st.warning(" · ".join(prediction.abstention_reasons))
        risk_frame = pd.DataFrame(
            {
                "risk": [
                    "Polarization",
                    "Thermal stress",
                    "Efficiency instability",
                    "Curve shift",
                    "OOD",
                ],
                "score": [
                    diagnosis.polarization_risk,
                    diagnosis.thermal_stress,
                    diagnosis.efficiency_instability,
                    diagnosis.curve_shift,
                    diagnosis.ood_score,
                ],
            }
        )
        st.plotly_chart(
            px.bar(risk_frame, x="score", y="risk", orientation="h", range_x=[0, 1]),
            use_container_width=True,
        )
        st.write("Nearest reference cells:", ", ".join(diagnosis.nearest_cells) or "none")
        for statement in diagnosis.statements:
            st.write(f"- {statement}")
        st.caption(prediction.coverage_note)

with strategy:
    st.subheader("Simulated policy trade-offs")
    if st.button("Evaluate policies at selected temperature", use_container_width=True):
        st.session_state["policies"] = engine.policies(
            selected_cell, ambient_temperature_c=ambient_temperature, initial_soc=0.10
        )
    policy_response = st.session_state.get("policies")
    if policy_response is None:
        st.info("Choose a temperature and evaluate the policy library.")
    else:
        rows: list[dict[str, object]] = []
        for evaluated in policy_response.policies:
            metrics = evaluated.trajectory.metrics
            rows.append(
                {
                    "policy_id": evaluated.policy.policy_id,
                    "family": evaluated.policy.family.value,
                    "decision": evaluated.safety.decision.value,
                    "pareto": evaluated.pareto_optimal,
                    "charge_time_min": metrics.charge_time_min if metrics else None,
                    "degradation_proxy": metrics.degradation_proxy if metrics else None,
                    "max_temperature_c": metrics.max_temperature_c if metrics else None,
                    "energy_loss_wh": metrics.energy_loss_wh if metrics else None,
                }
            )
        policy_frame = pd.DataFrame(rows)
        st.dataframe(policy_frame, use_container_width=True, hide_index=True)
        plot_frame = policy_frame.dropna(subset=["charge_time_min", "degradation_proxy"])
        if not plot_frame.empty:
            st.plotly_chart(
                px.scatter(
                    plot_frame,
                    x="charge_time_min",
                    y="degradation_proxy",
                    color="max_temperature_c",
                    symbol="decision",
                    hover_name="family",
                    size_max=18,
                    title="Safety-filtered objective space",
                ),
                use_container_width=True,
            )
        st.write("Pareto front:", ", ".join(policy_response.pareto_front) or "none")

with safety:
    st.subheader("Deterministic safety decisions")
    policy_response = st.session_state.get("policies")
    if policy_response is None:
        st.info("Evaluate policies in Strategy Lab first.")
    else:
        for evaluated in policy_response.policies:
            label = f"{evaluated.policy.family.value} · {evaluated.safety.decision.value}"
            expanded = evaluated.safety.decision != SafetyDecision.ALLOW
            with st.expander(label, expanded=expanded):
                st.code(evaluated.safety.safety_case_hash, language=None)
                if not evaluated.safety.violations:
                    st.success("All enforced checks passed in this simulation.")
                for violation in evaluated.safety.violations:
                    st.error(
                        f"{violation.constraint}: {violation.value} (limit {violation.limit}) — "
                        f"{violation.message}"
                    )

with reveal:
    st.subheader("Controlled synthetic reveal")
    st.write(
        "This token-gated action tests runtime authorization and appends validation evidence; "
        "it never overwrites the original prediction. The synthetic outcome is public source data."
    )
    evaluator_token = st.text_input("Evaluator token", type="password")
    if st.button("Run controlled synthetic reveal", type="primary"):
        try:
            st.session_state["reveal"] = engine.reveal(selected_cell, evaluator_token)
        except RevealAuthorizationError:
            st.error("Invalid evaluator token. The rejected attempt was recorded without storing it.")
        except AlreadyRevealedError:
            st.error("This cell has already been revealed; a second reveal is not allowed.")
    if result := st.session_state.get("reveal"):
        actual, covered, error = st.columns(3)
        actual.metric("Actual lifetime", f"{result['actual_cycle_life']} cycles")
        covered.metric("Interval covered", "YES" if result["covered"] else "NO")
        error.metric("Absolute error", f"{result['absolute_error']:.1f} cycles")
        st.json(result, expanded=False)

with evidence:
    st.subheader("Append-only Evidence Ledger")
    records = engine.evidence()
    st.write(f"Records: {len(records)} · chain valid: {engine.ledger.verify_chain()}")
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    st.download_button(
        "Download evidence JSON",
        data=json.dumps(records, ensure_ascii=False, indent=2),
        file_name="batteryguard-evidence.json",
        mime="application/json",
    )
