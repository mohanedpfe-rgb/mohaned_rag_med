from __future__ import annotations

import re
from typing import Any


_HIGH_RISK_PATTERNS = (
    r"\bdiagnos(?:e|is|ing|ed)\b", r"\bdiagnostic(?:s|que)?\b", r"\bdiagnostic(?:o|a)?\b",
    r"\bprescri(?:be|bed|bing|ption)\b", r"\bprescription\b", r"\bordonnance\b", r"\bprescrire\b",
    r"\bdos(?:e|age|ing)\b", r"\bdose\b", r"\bdosage\b", r"\bmg\s*/\s*kg\b", r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|mL|µg|iu|%)\b",
    r"\bcontraindicat(?:ed|ion)\b", r"\bcontre[- ]indiqu", r"\bdrug interaction\b", r"\binteraction médicamenteuse\b",
    r"\bemergency\b", r"\burgence\b", r"\bshould i (?:take|stop|start|use|increase|decrease)\b",
    r"\bwhat medication\b", r"\bwhich medication\b", r"\bwhat drug\b", r"\bmedication\b", r"\bmedicament\b", r"\bmédicament\b",
    r"\btreatment\b", r"\btraitement\b", r"\btherapy\b", r"\bthérapie\b", r"\bsymptom\b", r"\bsymptôme\b",
    r"\btake\b.{0,30}\btablet|\bcapsule|\binject|\binfusion\b", r"\bstop\b.{0,30}\bmedication|drug|insulin\b",
    r"\bتشخيص\b", r"\bدواء\b", r"\bجرعة\b", r"\bالعلاج\b", r"\bمضاد\b", r"\bطوارئ\b",
)

_ACTIONABLE_PATTERNS = (
    r"\b(i|i'm|i am|my|mine|me)\b",
    r"\b(should i|can i|may i|do i need|what should i|how should i)\b",
    r"\b(take|stop|start|increase|decrease|skip|replace)\b.{0,45}\b(medication|medicine|drug|tablet|capsule|insulin|dose|treatment)\b",
    r"\bprescri(?:be|bed|bing|ption)\b|\bprescription\b|\bordonnance\b|\bprescrire\b",
    r"\bdiagnos(?:e|is|ing|ed)\b.{0,45}\b(me|my|i|patient)\b",
    r"\b(urgence|emergency)\b.{0,60}\b(what|should|do|take|stop)\b",
)


def is_actionable_medical_query(question: str) -> bool:
    value = str(question or "").casefold()
    return any(re.search(pattern, value) for pattern in _ACTIONABLE_PATTERNS)


def is_high_risk_medical_query(question: str) -> bool:
    """Classify clinically sensitive queries without making every educational mention maximally restrictive."""
    value = str(question or "").casefold()
    return any(re.search(pattern, value) for pattern in _HIGH_RISK_PATTERNS)


def _effective_threshold(question: str, settings: Any) -> float:
    configured = float(getattr(settings, "medical_high_risk_evidence_threshold", 0.80))
    # Patient-specific/actionable requests remain at the strict threshold.
    if is_actionable_medical_query(question):
        return max(0.80, min(1.0, configured))
    # Educational questions still require grounding + citations, but should not be
    # rejected merely because the evidence score is below the clinical-action threshold.
    return max(0.60, min(0.85, configured * 0.85))


def apply_medical_safety_policy(question: str, result: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Deterministic post-generation medical safety boundary; never claims clinical validation."""
    result = dict(result or {})
    high_risk = is_high_risk_medical_query(question)
    actionable = is_actionable_medical_query(question)
    result.setdefault("medical_safety", {})
    result["medical_safety"].update({"high_risk_query": high_risk, "actionable_query": actionable, "policy_version": "2.1", "clinical_validation_claim": False})
    if not high_risk:
        result["medical_safety"]["decision"] = "STANDARD_GROUNDED_RESPONSE"
        return result
    confidence = float((result.get("confidence") or {}).get("evidence_confidence", 0.0) or 0.0)
    grounding = result.get("grounding") or result.get("certification", {}).get("grounding") or {}
    contradiction = result.get("contradiction_report") or result.get("certification", {}).get("contradiction") or {}
    grounding_ok = bool(grounding.get("allow", False))
    contradiction_free = not bool(contradiction.get("has_contradiction", False))
    citations = result.get("citations") or []
    threshold = _effective_threshold(question, settings)
    allowed = confidence >= threshold and grounding_ok and contradiction_free and bool(citations)
    result["medical_safety"].update({"decision": "ALLOW_WITH_EVIDENCE" if allowed else "ABSTAIN_HIGH_RISK", "required_evidence_confidence": threshold, "evidence_confidence": confidence, "grounding_ok": grounding_ok, "contradiction_free": contradiction_free, "citation_count": len(citations)})
    if not allowed:
        result["status"] = "MEDICAL_SAFETY_ABSTAIN"
        result["answer"] = "I can't safely provide a clinical recommendation from the available indexed evidence. The evidence did not meet the required medical verification threshold."
        result["citations"] = []
    return result


__all__ = ["is_high_risk_medical_query", "is_actionable_medical_query", "apply_medical_safety_policy"]
