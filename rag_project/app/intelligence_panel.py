from __future__ import annotations

from typing import Any

import streamlit as st


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _show_list(label: str, values: Any, limit: int = 16) -> None:
    vals = [str(x) for x in (values or ()) if str(x).strip()]
    if vals:
        st.write(f"**{label}:** " + ", ".join(vals[:limit]))


def _status_icon(status: str) -> str:
    return {
        "SUPPORTED": "✅",
        "ENTAILED": "✅",
        "PARTIAL": "🟡",
        "PARTIALLY_ENTAILED": "🟡",
        "WEAK": "🟠",
        "UNSUPPORTED": "❌",
        "CONTRADICTED": "❌",
        "NUMERIC_MISMATCH": "❌",
        "NOT_ENTAILED": "❌",
    }.get(str(status).upper(), "•")


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def render_intelligence_panel(result: dict[str, Any]) -> None:
    """Render the runtime intelligence contract exposed by the canonical answer pipeline."""
    result = result if isinstance(result, dict) else {}
    with st.expander("Intelligence pipeline · 5 phases", expanded=True):
        authority = result.get("pipeline_authority") or (
            result.get("phase_implementation") or {}
        ).get("authority") or "—"
        st.caption(f"Authoritative pipeline: `{authority}`")

        plan = result.get("phase_plan") or result.get("query_analysis") or {}
        phase_state = result.get("phases") or {}
        implementation = result.get("phase_implementation") or {}

        phase_keys = {
            "P1": "phase_1_query_understanding",
            "P2": "phase_2_retrieval_precision",
            "P3": "phase_3_two_stage_generation",
            "P4": "phase_4_verification",
            "P5": "phase_5_intelligence_visibility",
        }
        labels = (("P1", "Query"), ("P2", "Retrieval"), ("P3", "Generation"), ("P4", "Verification"), ("P5", "UI"))
        cols = st.columns(5)
        for col, (code, label) in zip(cols, labels):
            with col:
                key = phase_keys[code]
                runtime = implementation.get(key)
                value = runtime.get("status") if isinstance(runtime, dict) else runtime
                value = value or phase_state.get(key) or "unknown"
                st.metric(f"{code} {label}", str(value).upper())

        st.divider()
        _show_list("Intent", [plan.get("intent", "")])
        _show_list("Entities", plan.get("entities"))
        _show_list("Sub-questions", plan.get("sub_questions"))
        _show_list("Rewritten queries", plan.get("rewritten_queries"))
        _show_list("Must contain", plan.get("must_contain"))
        rewritten_question = result.get("rewritten_question")
        if rewritten_question:
            st.code(str(rewritten_question), language=None)
            st.caption("Runtime rewritten question used for retrieval/context construction.")
        st.write(
            f"**Ambiguity:** {plan.get('ambiguity', '—')} · "
            f"**Planner:** {plan.get('planner_source', '—')} · "
            f"**Planner confidence:** {_pct(plan.get('planner_confidence'))}"
        )
        st.write(
            f"**Answer shape:** {plan.get('answer_shape', '—')} · "
            f"**Numeric:** {bool(plan.get('needs_numeric'))} · "
            f"**Table:** {bool(plan.get('needs_table'))} · "
            f"**Figure:** {bool(plan.get('needs_figure'))} · "
            f"**Multi-hop:** {bool(plan.get('needs_multi_hop'))}"
        )

        retrieval = result.get("adaptive_retrieval") or result.get("adaptive_retrieval_budget") or {}
        st.write(
            f"**Adaptive retrieval:** stage {retrieval.get('stage', '—')} · "
            f"queries {retrieval.get('queries', '—')} · "
            f"candidate K {retrieval.get('candidate_k', retrieval.get('budget', '—'))} · "
            f"final hits {retrieval.get('final_hits', '—')} · "
            f"escalated={bool(retrieval.get('escalated'))}"
        )
        _show_list("Escalation reasons", retrieval.get("reasons"))

        terms = result.get("medical_term_layer") or {}
        _show_list("Detected medical terms", terms.get("terms"))
        _show_list("Drugs", terms.get("drug_like"))
        _show_list("Condition-like terms", terms.get("condition_like"))
        _show_list("Abbreviations", terms.get("abbreviations"))
        _show_list("Units", terms.get("units"))

        extractive = result.get("extractive_stage") or {}
        two_stage = result.get("two_stage_synthesis") or {}
        st.write(
            f"**Extractive stage:** {extractive.get('sentence_count', 0)} evidence sentences · "
            f"supported={bool(extractive.get('supported'))}"
        )
        st.write(
            f"**Synthesis stage:** required={bool(two_stage.get('required'))} · "
            f"attempted={bool(two_stage.get('attempted'))} · "
            f"used={bool(two_stage.get('used'))} · "
            f"fallback={bool(two_stage.get('fallback'))} · "
            f"verified={not bool((two_stage.get('verification') or {}).get('blocked'))} · "
            f"temperature={two_stage.get('temperature', result.get('generation_temperature', '—'))}"
        )

        calibration = result.get("confidence_calibration") or {}
        confidence = result.get("confidence") or {}
        calibrated = calibration.get("calibrated", confidence.get("evidence_confidence", 0))
        st.write(
            f"**Calibrated confidence:** {_pct(calibrated)} · "
            f"**level:** {calibration.get('level', confidence.get('level', '—'))}"
        )
        _show_list("Confidence reasons", calibration.get("reasons"))

        matrix = result.get("evidence_claim_matrix") or []
        checks = result.get("claim_checks") or result.get("claims") or []
        if matrix:
            st.markdown("**Claim → evidence matrix**")
            for raw_row in matrix[:16]:
                row = _row_dict(raw_row)
                status = row.get("status", "")
                evidence = row.get("evidence") or row.get("evidence_spans") or row.get("spans") or []
                st.write(
                    f"{_status_icon(status)} **{status or 'UNKNOWN'}** · "
                    f"{row.get('claim', '')} · support {_pct(row.get('support', 0))} · "
                    f"evidence spans {len(evidence)}"
                )
        elif checks:
            st.markdown("**Claim support**")
            for raw_row in checks[:16]:
                row = _row_dict(raw_row)
                status = row.get("status", "")
                sources = row.get("sources", ())
                st.write(
                    f"{_status_icon(status)} **{status or 'UNKNOWN'}** · "
                    f"{row.get('claim', '')} · support {_pct(row.get('support', row.get('entailment', 0)))} · "
                    f"sources {', '.join(map(str, sources)) if sources else '—'}"
                )

        reasoning = result.get("advanced_reasoning") or {}
        _show_list("Reasoning blocked reasons", reasoning.get("blocked_reasons"))
        st.write(
            f"**Reasoning:** mode={reasoning.get('mode', '—')} · "
            f"depth={reasoning.get('depth', '—')} · "
            f"path support={_pct(reasoning.get('path_support', 0))} · "
            f"source agreement={_pct(reasoning.get('source_agreement', 0))} · "
            f"safety conflict={_pct(reasoning.get('safety_conflict', 0))}"
        )

        contract = result.get("production_contract") or {}
        st.write(
            f"**Pipeline contract:** features={contract.get('feature_count', '—')} · "
            f"all resolved={contract.get('all_features_resolved', contract.get('all_resolved', '—'))}"
        )
        phase5 = implementation.get("phase_5_intelligence_visibility") or {}
        st.write(
            f"**UI visibility contract:** signals present={phase5.get('signals_present', False)} · "
            f"authority={phase5.get('authority', authority)}"
        )

        status = str(result.get("status", "")).upper()
        reasons = reasoning.get("blocked_reasons") or result.get("abstention_reasons") or []
        if result.get("abstained") or status in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "GENERATION_ABSTAIN"}:
            reason_text = f" Reasons: {', '.join(map(str, reasons))}" if reasons else ""
            st.warning("The system withheld unsupported content." + reason_text)
