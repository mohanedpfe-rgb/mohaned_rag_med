from __future__ import annotations

from typing import Any


def build_visibility_contract(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize the mandatory Phase-5 intelligence signals for the UI."""
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

    abstention_present = any(
        key in payload for key in ("abstention_reasons", "abstained", "status")
    ) or "blocked_reasons" in (payload.get("advanced_reasoning") or {})
    abstention = payload.get("abstention_reasons")
    if abstention is None:
        abstention = (payload.get("advanced_reasoning") or {}).get("blocked_reasons") or []

    required = {
        "intent": bool(str(plan.get("intent") or "").strip()),
        "entities": "entities" in plan,
        "rewritten_question": "rewritten_question" in payload,
        "claim_support_matrix": matrix_present,
        "calibrated_confidence": "confidence_calibration" in payload,
        "abstention_reason": abstention_present,
    }
    visible_count = sum(1 for present in required.values() if present)

    calibrated = calibration.get("calibrated", confidence.get("evidence_confidence"))
    level = calibration.get("level", confidence.get("level", "none"))
    reasons = [str(item).strip() for item in (abstention or ()) if str(item).strip()]

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
