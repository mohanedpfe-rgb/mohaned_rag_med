from __future__ import annotations

from typing import Any

import streamlit as st


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def render_production_contract_panel(result: dict[str, Any]) -> None:
    """Render request/evidence/verification contract signals without exposing internals as claims."""
    result = result if isinstance(result, dict) else {}
    context = result.get("request_context") or {}
    evidence = result.get("evidence_bundle") or {}
    envelope = result.get("answer_envelope") or {}
    confidence = result.get("confidence_breakdown") or result.get("confidence_calibration") or {}
    matrix = result.get("claim_matrix_summary") or {}

    with st.expander("Production contract · request/evidence/verification", expanded=False):
        st.caption(f"Contract: `{result.get('contract_version', '—')}` · Request: `{result.get('request_id', '—')}`")
        a, b, c = st.columns(3)
        with a:
            st.metric("Request state", str(envelope.get("status", result.get("contract_status", "—"))).upper())
        with b:
            st.metric("Evidence hits", str(evidence.get("hit_count", 0)))
        with c:
            st.metric("Final confidence", _pct(confidence.get("final", confidence.get("calibrated", 0))))

        st.write(
            f"**Canonical query:** {context.get('canonical_question', result.get('rewritten_question', '—'))}"
        )
        st.write(
            f"**Complexity:** `{context.get('complexity', '—')}` · **Intent:** `{context.get('intent', '—')}` · "
            f"**Follow-up:** {bool(context.get('is_followup'))} · **Conversation used:** {bool(context.get('conversation_used'))}"
        )
        st.write(
            f"**Evidence:** top {_pct(evidence.get('top_score', 0))} · mean {_pct(evidence.get('mean_score', 0))} · "
            f"entity coverage {_pct(evidence.get('entity_coverage', 0))} · missing entities {len(evidence.get('missing_entities', ()))}"
        )
        st.write(
            f"**Confidence signals:** retrieval {_pct(confidence.get('retrieval', 0))} · evidence {_pct(confidence.get('evidence_quality', 0))} · "
            f"entailment {_pct(confidence.get('entailment', 0))} · verification {_pct(confidence.get('verification', 0))} · "
            f"contradiction {_pct(confidence.get('contradiction', 0))}"
        )
        st.write(
            f"**Claim matrix:** {matrix.get('claim_count', 0)} claims · {matrix.get('entailed_claims', 0)} entailed · "
            f"{matrix.get('blocked_claims', 0)} blocked · supported ratio {_pct(matrix.get('supported_ratio', 0))}"
        )
        if envelope.get("status") == "abstain":
            st.warning(
                "Controlled abstention: the answer is a state, not a medical claim. "
                f"Reason: {envelope.get('control_reason') or envelope.get('verification_reason') or '—'}"
            )


__all__ = ["render_production_contract_panel"]
