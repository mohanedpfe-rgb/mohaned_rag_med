from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.med_evidence_pro import SafetyGate
from rag_project.runtime_functionality_scope_fix import _wrap_safety_scope


def test_phrase_based_medical_query_is_not_rejected_as_out_of_scope():
    original = lambda self, query, context: SimpleNamespace(action="ABSTAIN", reason="outside_medical_scope", scope_confidence=0.25)
    decision = _wrap_safety_scope(original)(SafetyGate(), "What are the side effects of aspirin?", "")
    assert decision.action == "PROCEED"
    assert decision.reason == "in_scope"


def test_non_medical_query_remains_out_of_scope():
    original = lambda self, query, context: SimpleNamespace(action="ABSTAIN", reason="outside_medical_scope", scope_confidence=0.25)
    decision = _wrap_safety_scope(original)(SafetyGate(), "How do I repair my laptop?", "")
    assert decision.action == "ABSTAIN"
