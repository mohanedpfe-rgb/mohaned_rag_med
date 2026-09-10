from __future__ import annotations

from typing import Any

import streamlit as st

from rag_project.intelligence.ui_visibility_contract import build_visibility_contract


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
        "SUPPORTED": "✅", "ENTAILED": "✅", "PARTIAL": "🟡", "PARTIALLY_ENTAILED": "🟡",
        "WEAK": "🟠", "UNSUPPORTED": "❌", "CONTRADICTED": "❌", "NUMERIC_MISMATCH": "❌",
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


def _render_phase5_transparency(result: dict[str, Any]) -> None:
    """Render the mandatory Phase-5 signals as a first-class, human-readable contract."""
    visibility = build_visibility_contract(result)
    signals = visibility["signals"]
    st.markdown("### Decision transparency")
    state = "READY" if visibility["signals_present"] else "INCOMPLETE"
    st.caption(
        f"Phase 5 visibility: **{state}** · "
        f"{visibility['visible_signal_count']}/{visibility['required_signal_count']} required signals exposed"
    )

    left, right = st.columns(2)
    with left:
        st.write(f"**Intent:** {visibility['intent']} {'✅' if signals['intent'] else '⚠️'}")
        entities = ", ".join(map(str, visibility["entities"][:16])) or "—"
        st.write(f"**Entities:** {entities} {'✅' if signals['entities'] else '⚠️'}")
        st.write(f"**Calibrated confidence:** {_pct(visibility['calibrated_confidence'])} · level `{visibility['confidence_level']}` {'✅' if signals['calibrated_confidence'] else '⚠️'}")
        reasons = visibility["abstention_reasons"]
        st.write(f"**Abstention reason:** {'; '.join(reasons) if reasons else 'None — answer was not withheld for an abstention reason.'} ✅")
    with right:
        st.write("**Rewritten question:**")
        st.code(visibility["rewritten_question"], language=None)
        matrix = visibility["claim_support_matrix"]
        st.write(f"**Claim support matrix:** {len(matrix)} claim row(s) {'✅' if signals['claim_support_matrix'] else '⚠️'}")
        if matrix:
            for raw_row in matrix[:8]:
                row = _row_dict(raw_row)
                status = row.get("status", "UNKNOWN")
                st.write(
                    f"{_status_icon(status)} **{status}** · {row.get('claim', '')} · "
                    f"support {_pct(row.get('support', row.get('entailment', 0)))}"
                )


def render_intelligence_panel(result: dict[str, Any]) -> None:
    """Render the runtime intelligence contract exposed by the canonical answer pipeline."""
    result = result if isinstance(result, dict) else {}
    with st.expander("Intelligence pipeline · 5 phases", expanded=True):
        authority = result.get("pipeline_authority") or (result.get("phase_implementation") or {}).get("authority") or "—"
        st.caption(f"Authoritative pipeline: `{authority}`")

        plan = result.get("phase_plan") or result.get("query_analysis") or {}
        phase_state = result.get("phases") or {}
        implementation = result.get("phase_implementation") or {}
        phase_keys = {"P1":"phase_1_query_understanding","P2":"phase_2_retrieval_precision","P3":"phase_3_two_stage_generation","P4":"phase_4_verification","P5":"phase_5_intelligence_visibility"}
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
        _render_phase5_transparency(result)
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
        st.write(f"**Ambiguity:** {plan.get('ambiguity', '—')} · **Planner:** {plan.get('planner_source', '—')} · **Planner confidence:** {_pct(plan.get('planner_confidence'))}")
        st.write(f"**Answer shape:** {plan.get('answer_shape', '—')} · **Numeric:** {bool(plan.get('needs_numeric'))} · **Table:** {bool(plan.get('needs_table'))} · **Figure:** {bool(plan.get('needs_figure'))} · **Multi-hop:** {bool(plan.get('needs_multi_hop'))}")

        retrieval = result.get("adaptive_retrieval") or result.get("adaptive_retrieval_budget") or {}
        st.write(f"**Adaptive retrieval:** stage {retrieval.get('stage', '—')} · queries {retrieval.get('queries', '—')} · candidate K {retrieval.get('candidate_k', retrieval.get('budget', '—'))} · final hits {retrieval.get('final_hits', '—')} · escalated={bool(retrieval.get('escalated'))}")
        _show_list("Escalation reasons", retrieval.get("reasons"))

        terms = result.get("medical_term_layer") or {}
        _show_list("Detected medical terms", terms.get("terms"))
        _show_list("Drugs", terms.get("drug_like"))
        _show_list("Condition-like terms", terms.get("condition_like"))
        _show_list("Abbreviations", terms.get("abbreviations"))
        _show_list("Units", terms.get("units"))

        entity_report = result.get("entity_coverage") or {}
        if entity_report:
            st.write(f"**Entity coverage:** {_pct(entity_report.get('coverage'))} direct · {_pct(entity_report.get('partial_coverage'))} partial-adjusted · query entities {entity_report.get('entity_count', 0)}")
            _show_list("Missing entities", entity_report.get("missing"))
            _show_list("Partially matched entities", entity_report.get("partial"))

        extractive = result.get("extractive_stage") or {}
        two_stage = result.get("two_stage_synthesis") or {}
        st.write(f"**Extractive stage:** {extractive.get('sentence_count', 0)} evidence sentences · supported={bool(extractive.get('supported'))}")
        st.write(f"**Synthesis stage:** required={bool(two_stage.get('required'))} · attempted={bool(two_stage.get('attempted'))} · used={bool(two_stage.get('used'))} · fallback={bool(two_stage.get('fallback'))} · verified={not bool((two_stage.get('verification') or {}).get('blocked'))} · temperature={two_stage.get('temperature', result.get('generation_temperature', '—'))}")

        calibration = result.get("confidence_calibration") or {}
        confidence = result.get("confidence") or {}
        calibrated = calibration.get("calibrated", confidence.get("evidence_confidence", 0))
        st.write(f"**Calibrated confidence:** {_pct(calibrated)} · **level:** {calibration.get('level', confidence.get('level', '—'))}")
        _show_list("Confidence reasons", calibration.get("reasons"))

        matrix = result.get("evidence_claim_matrix") or []
        checks = result.get("final_claim_checks") or result.get("claim_checks") or result.get("claims") or []
        if matrix:
            st.markdown("**Final claim → evidence matrix**")
            for raw_row in matrix[:16]:
                row = _row_dict(raw_row)
                status = row.get("status", "")
                evidence = row.get("evidence") or row.get("evidence_spans") or row.get("spans") or []
                st.write(f"{_status_icon(status)} **{status or 'UNKNOWN'}** · {row.get('claim', '')} · support {_pct(row.get('support', 0))} · evidence spans {len(evidence)}")
        elif checks:
            st.markdown("**Final claim support**")
            for raw_row in checks[:16]:
                row = _row_dict(raw_row)
                status = row.get("status", "")
                sources = row.get("sources", ())
                st.write(f"{_status_icon(status)} **{status or 'UNKNOWN'}** · {row.get('claim', '')} · support {_pct(row.get('support', row.get('entailment', 0)))} · sources {', '.join(map(str, sources)) if sources else '—'}")

        final_verification = result.get("final_verification") or {}
        if final_verification:
            gate = "PASS" if final_verification.get("allow") else "BLOCKED"
            icon = "✅" if final_verification.get("allow") else "❌"
            st.write(f"**Final answer gate:** {icon} {gate} · claims={final_verification.get('claim_count', 0)} · blocked={final_verification.get('blocked_claims', 0)} · support ratio={_pct(final_verification.get('supported_ratio', 0))} · matrix fully entailed={bool(final_verification.get('matrix_all_entailed'))} · reason={final_verification.get('reason', '—')}")

        reasoning = result.get("advanced_reasoning") or {}
        _show_list("Reasoning blocked reasons", reasoning.get("blocked_reasons"))
        st.write(f"**Reasoning:** mode={reasoning.get('mode', '—')} · depth={reasoning.get('depth', '—')} · path support={_pct(reasoning.get('path_support', 0))} · source agreement={_pct(reasoning.get('source_agreement', 0))} · safety conflict={_pct(reasoning.get('safety_conflict', 0))}")

        contract = result.get("production_contract") or {}
        st.write(f"**Pipeline contract:** features={contract.get('feature_count', '—')} · all resolved={contract.get('all_features_resolved', contract.get('all_resolved', '—'))}")
        phase5 = implementation.get("phase_5_intelligence_visibility") or {}
        st.write(f"**UI visibility contract:** signals present={phase5.get('signals_present', False)} · authority={phase5.get('authority', authority)} · canonical executed={bool(implementation.get('canonical_pipeline_executed'))}")

        status = str(result.get("status", "")).upper()
        reasons = reasoning.get("blocked_reasons") or result.get("abstention_reasons") or []
        if result.get("abstained") or status in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "GENERATION_ABSTAIN", "MEDICAL_SAFETY_ABSTAIN"}:
            reason_text = f" Reasons: {', '.join(map(str, reasons))}" if reasons else ""
            st.warning("The system withheld unsupported content." + reason_text)


__all__ = ["render_intelligence_panel"]
