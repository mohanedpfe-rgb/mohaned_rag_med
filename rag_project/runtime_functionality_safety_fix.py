"""Functionality correction for negated emergency symptom phrases."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any


def _emergency_term_is_negated(query: str, term: str) -> bool:
    text = str(query or "").casefold()
    needle = str(term or "").casefold()
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return False
        prefix = text[max(0, index - 48):index]
        if re.search(
            r"(?:\b(?:no|not|without|denies|denied|negative for|free of)\b|\b(?:لا|لم|ليس|ليست|بدون|دون)\b)"
            r"(?:[\s,:;()/-]+\w+){0,4}[\s,:;()/-]*$",
            prefix,
            flags=re.I | re.UNICODE,
        ):
            start = index + len(needle)
            continue
        return False


def _wrap_safety_check(original: Any):
    from rag_project.intelligence.med_evidence_pro import EMERGENCY_TERMS

    def wrapped(self: Any, query: str, context: str = ""):
        decision = original(self, query, context)
        if not getattr(decision, "emergency", False):
            return decision
        normalized = str(query or "")
        active_emergency = any(
            term in normalized.casefold() and not _emergency_term_is_negated(normalized, term)
            for term in EMERGENCY_TERMS
        )
        if active_emergency:
            return decision
        return replace(decision, reason="in_scope", emergency=False)

    wrapped._functionality_safety_negation_fix = True
    return wrapped


def install() -> None:
    from rag_project.intelligence.med_evidence_pro import SafetyGate

    original = SafetyGate.check
    if not getattr(original, "_functionality_safety_negation_fix", False):
        SafetyGate.check = _wrap_safety_check(original)


__all__ = ["_emergency_term_is_negated", "_wrap_safety_check", "install"]
