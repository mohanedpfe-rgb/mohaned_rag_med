from __future__ import annotations

from typing import Any

import streamlit as st
from rag_project.intelligence.ui_visibility_contract import build_visibility_contract


def _render_phase5_transparency(result: dict[str, Any]) -> None:
    result = {**result, **build_visibility_contract(result)}
    visibility = result.get("phase_5_intelligence_visibility", result)
    st.caption(f"Intent: {visibility.get('intent', '—')}")
    st.caption(f"Entities: {visibility.get('entities', [])}")
    st.caption(f"Rewritten question: {visibility.get('rewritten_question', '—')}")
    st.caption(f"Claim support matrix: {visibility.get('claim_support_matrix', visibility.get('evidence_claim_matrix', []))}")
    st.caption(f"Calibrated confidence: {visibility.get('calibrated_confidence', visibility.get('confidence_calibration', '—'))}")
    st.caption(f"Abstention reason: {visibility.get('abstention_reasons', [])}")
    st.caption(f"Phase 5 visibility: {visibility.get('signals_present', True)}")


def render_intelligence_panel(result: dict[str, Any] | None) -> None:
    result = dict(result or {})
    with st.expander("Intelligence pipeline · 5 phases", expanded=True):
        st.subheader("Decision transparency")
        _render_phase5_transparency(result)
        for key in (
            "pipeline_authority", "phase_plan", "evidence_claim_matrix",
            "confidence_calibration", "two_stage_synthesis", "adaptive_retrieval",
            "medical_term_layer", "entity_coverage", "final_claim_checks",
            "final_verification", "advanced_reasoning", "production_contract",
        ):
            if key in result:
                st.caption(f"{key}: {result[key]}")


__all__ = ["render_intelligence_panel"]
