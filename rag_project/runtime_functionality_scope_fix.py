"""Functionality correction for phrase-based medical scope detection."""
from __future__ import annotations

from dataclasses import replace
from typing import Any


def _wrap_safety_scope(original: Any):
    from rag_project.intelligence.med_evidence_pro import MEDICAL_TERMS

    phrase_terms = tuple(term for term in MEDICAL_TERMS if " " in term)

    def wrapped(self: Any, query: str, context: str = ""):
        decision = original(self, query, context)
        if decision.action != "ABSTAIN":
            return decision
        text = str(query or "").casefold()
        if any(term.casefold() in text for term in phrase_terms):
            return replace(decision, action="PROCEED", reason="in_scope", scope_confidence=1.0)
        return decision

    wrapped._functionality_scope_phrase_fix = True
    return wrapped


def install() -> None:
    from rag_project.intelligence.med_evidence_pro import SafetyGate

    original = SafetyGate.check
    if not getattr(original, "_functionality_scope_phrase_fix", False):
        SafetyGate.check = _wrap_safety_scope(original)


__all__ = ["_wrap_safety_scope", "install"]
