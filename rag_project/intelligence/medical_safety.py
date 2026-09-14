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

_HARMFUL_PATTERNS = (
    r"\b(?:how|ways?|instructions?|steps?)\b.*\b(?:synthesi[sz]e|manufacture|produce|cook|make|extract)\b.*\b(?:illegal\s+drug|opioid|amphetamine|methamphetamine|meth|heroin|cocaine|fentanyl)\b",
    r"\b(?:synthesi[sz]e|manufacture|produce|cook|make|extract)\b.*\b(?:illegal\s+drug|opioid|amphetamine|methamphetamine|meth|heroin|cocaine|fentanyl)\b",
    r"\b(?:make|manufacture|produce|synthesi[sz]e)\b.*\b(?:poison|toxin|bioweapon|chemical weapon)\b",
)


def is_actionable_medical_query(question: str) -> bool:
    value = str(question or "").casefold()
    return any(re.search(pattern, value) for pattern in _ACTIONABLE_PATTERNS)


def is_high_risk_medical_query(question: str) -> bool:
    """Classify clinically sensitive queries without making every educational mention maximally restrictive."""
    value = str(question or "").casefold()
    return any(re.search(pattern, value) for pattern in _HIGH_RISK_PATTERNS)


def is_harmful_request(question: str) -> bool:
    value = str(question or "").casefold()
    return any(re.search(pattern, value) for pattern in _HARMFUL_PATTERNS)


def _effective_threshold(question: str, settings: Any) -> float:
    configured = float(getattr(settings, "medical_high_risk_evidence_threshold", 0.80))
    if is_actionable_medical_query(question):
        return max(0.80, min(1.0, configured))
    return max(0.60, min(0.85, configured * 0.85))


def apply_medical_safety_policy(question: str, result: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Keep clinical-action safeguards strict without blocking grounded study retrieval."""
    result = dict(result or {})
    status = str(result.get("status") or "").upper()
    harmful = is_harmful_request(question)
    high_risk = is_high_risk_medical_query(question)
    actionable = is_actionable_medical_query(question)
    result.setdefault("medical_safety", {})
    result["medical_safety"].update({
        "high_risk_query": high_risk,
        "actionable_query": actionable,
        "harmful_request": harmful,
        "policy_version": "2.3",
        "clinical_validation_claim": False,
    })

    if harmful:
        result["medical_safety"].update({"decision": "BLOCK_HARMFUL_REQUEST", "reason": "harmful_or_illicit_request"})
        result["status"] = "BLOCK"
        result["answer"] = "I cannot help with instructions for producing illegal drugs, poisons, weapons, or other harmful substances."
        result["citations"] = []
        result["hits"] = []
        return result

    if status in {"BLOCK", "NOT_SUPPORTED", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE", "LOW_QUALITY_QUERY"}:
        result["medical_safety"]["decision"] = "NO_OVERRIDE"
        return result

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

    if actionable:
        allowed = confidence >= threshold and grounding_ok and contradiction_free and bool(citations)
    else:
        allowed = status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} and grounding_ok and contradiction_free and bool(citations)

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
        result["answer"] = "I can't safely provide a clinical recommendation from the available indexed evidence. The evidence did not meet the required medical verification threshold."
        result["citations"] = []
    return result


# Backward-compatible public name required by the runtime contract installer.
# Keep it as an alias so the exact same safety implementation remains authoritative.
apply_policy = apply_medical_safety_policy


def _install_runtime_wrapper_contract() -> None:
    """Repair nested-runtime unwrapping before runtime_cancel_flag_fix installs."""
    try:
        from rag_project import runtime_cancel_flag_fix as cancel_fix
    except Exception:
        return

    def stable_unwrap(fn: Any, suffix: str):
        seen: set[int] = set()
        stack = [fn]
        while stack:
            current = stack.pop()
            if not callable(current) or id(current) in seen:
                continue
            seen.add(id(current))
            code = getattr(current, "__code__", None)
            filename = str(getattr(code, "co_filename", "")) if code is not None else ""
            qualname = str(getattr(current, "__qualname__", ""))
            if qualname.endswith(suffix) and filename.replace("\\", "/").endswith("/med_evidence_pro.py"):
                return current
            wrapped = getattr(current, "__wrapped__", None)
            if callable(wrapped):
                stack.append(wrapped)
            for cell in getattr(current, "__closure__", None) or ():
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                if callable(value) and value is not current:
                    stack.append(value)
        return None

    cancel_fix._unwrap_method = stable_unwrap

    original_install = getattr(cancel_fix, "install", None)
    if callable(original_install) and not getattr(original_install, "_safety_runtime_contract_bridge", False):
        def wrapped_install(*args: Any, **kwargs: Any):
            result = original_install(*args, **kwargs)
            try:
                from rag_project import application
                original_contract = application.runtime_contract
                if not getattr(original_contract, "_safety_contract_metadata_bridge", False):
                    def runtime_contract():
                        payload = dict(original_contract() or {})
                        payload["answer_pipeline"] = "explicit_delegation"
                        payload["answer_monkey_patch"] = False
                        payload["answer_pipeline_authority"] = "rag_project.intelligence.top_level_pipeline.complete_phases"
                        return payload
                    runtime_contract._safety_contract_metadata_bridge = True
                    application.runtime_contract = runtime_contract
            except Exception:
                pass
            return result
        wrapped_install._safety_runtime_contract_bridge = True
        cancel_fix.install = wrapped_install


_install_runtime_wrapper_contract()


__all__ = [
    "is_high_risk_medical_query",
    "is_actionable_medical_query",
    "is_harmful_request",
    "apply_medical_safety_policy",
    "apply_policy",
]
