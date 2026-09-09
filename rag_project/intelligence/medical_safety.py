from __future__ import annotations

import re
from typing import Any


_HIGH_RISK_PATTERNS = (
    r"\bdiagnos(?:e|is|ing|ed)\b",
    r"\bprescri(?:be|bed|bing|ption)\b",
    r"\bdos(?:e|age|ing)\b",
    r"\bmg\s*/\s*kg\b",
    r"\bcontraindicat(?:ed|ion)\b",
    r"\bdrug interaction\b",
    r"\bemergency\b",
    r"\bshould i (?:take|stop|start)\b",
    r"\bwhat medication\b",
    r"\btreatment\b",
)


def is_high_risk_medical_query(question: str) -> bool:
    value = (question or "").casefold()
    return any(re.search(pattern, value) for pattern in _HIGH_RISK_PATTERNS)


def apply_medical_safety_policy(question: str, result: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Apply a deterministic safety gate after grounding and citation checks.

    The policy never invents medical advice. High-risk questions require stronger
    evidence and clean grounding; otherwise the answer is converted to an explicit
    evidence-only abstention. This is a safety boundary, not a claim of clinical
    validation or regulatory approval.
    """
    result = dict(result or {})
    high_risk = is_high_risk_medical_query(question)
    result.setdefault("medical_safety", {})
    result["medical_safety"].update({
        "high_risk_query": high_risk,
        "policy_version": "1.0",
        "clinical_validation_claim": False,
    })
    if not high_risk:
        result["medical_safety"]["decision"] = "STANDARD_GROUNDED_RESPONSE"
        return result

    confidence = float((result.get("confidence") or {}).get("evidence_confidence", 0.0) or 0.0)
    grounding = result.get("grounding") or result.get("certification", {}).get("grounding") or {}
    grounding_ok = bool(grounding.get("allow", False))
    contradiction = result.get("contradiction_report") or result.get("certification", {}).get("contradiction") or {}
    contradiction_free = not bool(contradiction.get("has_contradiction", False))
    citations = result.get("citations") or []
    threshold = float(getattr(settings, "medical_high_risk_evidence_threshold", 0.80))
    allowed = confidence >= threshold and grounding_ok and contradiction_free and bool(citations)
    result["medical_safety"].update({
        "decision": "ALLOW_WITH_EVIDENCE" if allowed else "ABSTAIN_HIGH_RISK",
        "required_evidence_confidence": threshold,
        "evidence_confidence": confidence,
        "grounding_ok": grounding_ok,
        "contradiction_free": contradiction_free,
        "citation_count": len(citations),
    })
    if not allowed:
        result["status"] = "MEDICAL_SAFETY_ABSTAIN"
        result["answer"] = (
            "I can't safely provide a clinical recommendation from the available indexed evidence. "
            "The evidence did not meet the high-risk medical verification threshold."
        )
        result["citations"] = []
    return result


__all__ = ["is_high_risk_medical_query", "apply_medical_safety_policy"]
