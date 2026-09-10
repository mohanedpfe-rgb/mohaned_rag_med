from __future__ import annotations

from typing import Any


def build_visibility_contract(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize the mandatory Phase-5 intelligence signals for the UI.

    Presence is based on the effective runtime signal, including supported
    compatibility fallbacks (for example ``confidence.evidence_confidence``),
    rather than requiring one particular producer key.
    """
    payload = result if isinstance(result, dict) else {}
    plan = payload.get("phase_plan") or payload.get("query_analysis") or {}
    calibration = payload.get("confidence_calibration") or {}
    confidence = payload.get("confidence") or {}

    matrix_present = any(
        key in payload
        for key in ("evidence_claim_matrix", "final_evidence_claim_matrix", "final_claim_checks")
    )
    matrix = payload.get("evidence_claim_matrix")
    if matrix is None:
        matrix = payload.get("final_evidence_claim_matrix")
    if matrix is None:
        matrix = payload.get("final_claim_checks") or []

    advanced_reasoning = payload.get("advanced_reasoning") or {}
    abstention = payload.get("abstention_reasons")
    if abstention is None:
        abstention = advanced_reasoning.get("blocked_reasons") or []
    abstention_present = any(
        key in payload for key in ("abstention_reasons", "abstained", "status")
    ) or "blocked_reasons" in advanced_reasoning

    calibrated = calibration.get("calibrated", confidence.get("evidence_confidence"))
    level = calibration.get("level", confidence.get("level", "none"))
    reasons = [str(item).strip() for item in (abstention or ()) if str(item).strip()]

    required = {
        "intent": bool(str(plan.get("intent") or "").strip()),
        "entities": "entities" in plan and isinstance(plan.get("entities"), (list, tuple)),
        "rewritten_question": bool(str(payload.get("rewritten_question") or "").strip()),
        "claim_support_matrix": matrix_present,
        "calibrated_confidence": calibrated is not None,
        "abstention_reason": abstention_present,
    }
    visible_count = sum(1 for present in required.values() if present)

    return {
        "intent": str(plan.get("intent") or "—"),
        "entities": list(plan.get("entities") or ()),
        "rewritten_question": str(payload.get("rewritten_question") or "—"),
        "claim_support_matrix": list(matrix or ()),
        "calibrated_confidence": calibrated,
        "confidence_level": str(level),
        "abstention_reasons": reasons,
        "signals": required,
        "signals_present": visible_count == len(required),
        "visible_signal_count": visible_count,
        "required_signal_count": len(required),
    }


__all__ = ["build_visibility_contract"]
