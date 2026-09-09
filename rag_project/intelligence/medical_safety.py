from __future__ import annotations

import re
from typing import Any


_HIGH_RISK_PATTERNS = (
    r"\bdiagnos(?:e|is|ing|ed)\b", r"\bdiagnostic(?:s|que)?\b", r"\bdiagnostic(?:o|a)?\b",
    r"\bprescri(?:be|bed|bing|ption)\b", r"\bprescription\b", r"\bordonnance\b", r"\bprescrire\b",
    r"\bdos(?:e|age|ing)\b", r"\bdose\b", r"\bdosage\b", r"\bmg\s*/\s*kg\b", r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|µg|iu|%)\b",
    r"\bcontraindicat(?:ed|ion)\b", r"\bcontre[- ]indiqu", r"\bdrug interaction\b", r"\binteraction médicamenteuse\b",
    r"\bemergency\b", r"\burgence\b", r"\bshould i (?:take|stop|start|use|increase|decrease)\b",
    r"\bwhat medication\b", r"\bwhich medication\b", r"\bwhat drug\b", r"\bmedication\b", r"\bmedicament\b", r"\bmédicament\b",
    r"\btreatment\b", r"\btraitement\b", r"\btherapy\b", r"\bthérapie\b", r"\bsymptom\b", r"\bsymptôme\b",
    r"\btake\b.{0,30}\btablet|\bcapsule|\binject|\binfusion\b", r"\bstop\b.{0,30}\bmedication|drug|insulin\b",
    r"\bتشخيص\b", r"\bدواء\b", r"\bجرعة\b", r"\bالعلاج\b", r"\bمضاد\b", r"\bطوارئ\b",
)


def is_high_risk_medical_query(question: str) -> bool:
    value = (question or "").casefold()
    return any(re.search(pattern, value) for pattern in _HIGH_RISK_PATTERNS)


def apply_medical_safety_policy(question: str, result: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Deterministic post-generation medical safety boundary; never claims clinical validation."""
    result = dict(result or {})
    high_risk = is_high_risk_medical_query(question)
    result.setdefault("medical_safety", {})
    result["medical_safety"].update({"high_risk_query": high_risk, "policy_version": "2.0", "clinical_validation_claim": False})
    if not high_risk:
        result["medical_safety"]["decision"] = "STANDARD_GROUNDED_RESPONSE"
        return result
    confidence = float((result.get("confidence") or {}).get("evidence_confidence", 0.0) or 0.0)
    grounding = result.get("grounding") or result.get("certification", {}).get("grounding") or {}
    contradiction = result.get("contradiction_report") or result.get("certification", {}).get("contradiction") or {}
    grounding_ok = bool(grounding.get("allow", False))
    contradiction_free = not bool(contradiction.get("has_contradiction", False))
    citations = result.get("citations") or []
    threshold = float(getattr(settings, "medical_high_risk_evidence_threshold", 0.80))
    allowed = confidence >= threshold and grounding_ok and contradiction_free and bool(citations)
    result["medical_safety"].update({"decision": "ALLOW_WITH_EVIDENCE" if allowed else "ABSTAIN_HIGH_RISK", "required_evidence_confidence": threshold, "evidence_confidence": confidence, "grounding_ok": grounding_ok, "contradiction_free": contradiction_free, "citation_count": len(citations)})
    if not allowed:
        result["status"] = "MEDICAL_SAFETY_ABSTAIN"
        result["answer"] = "I can't safely provide a clinical recommendation from the available indexed evidence. The evidence did not meet the high-risk medical verification threshold."
        result["citations"] = []
    return result


__all__ = ["is_high_risk_medical_query", "apply_medical_safety_policy"]
