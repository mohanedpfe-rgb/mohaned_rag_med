from dataclasses import dataclass

from rag_project.runtime_functionality_safety_fix import _wrap_safety_check


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    confidence_threshold: float
    emergency: bool
    real_patient: bool = False
    high_rigor: bool = False
    scope_confidence: float = 1.0


def test_negated_emergency_symptom_is_not_marked_emergency(monkeypatch):
    import rag_project.intelligence.med_evidence_pro as med

    def original(self, query, context=""):
        return Decision("PROCEED", "emergency_signal", 0.75, True)

    monkeypatch.setattr(med, "EMERGENCY_TERMS", ("chest pain",))
    wrapped = _wrap_safety_check(original)
    decision = wrapped(object(), "I do not have chest pain, what causes fatigue?")
    assert decision.emergency is False
    assert decision.reason == "in_scope"


def test_positive_emergency_signal_remains_emergency(monkeypatch):
    import rag_project.intelligence.med_evidence_pro as med

    def original(self, query, context=""):
        return Decision("PROCEED", "emergency_signal", 0.75, True)

    monkeypatch.setattr(med, "EMERGENCY_TERMS", ("chest pain",))
    wrapped = _wrap_safety_check(original)
    decision = wrapped(object(), "I have severe chest pain right now")
    assert decision.emergency is True
    assert decision.reason == "emergency_signal"


def test_negated_emergency_in_one_clause_does_not_hide_a_real_emergency(monkeypatch):
    import rag_project.intelligence.med_evidence_pro as med

    def original(self, query, context=""):
        return Decision("PROCEED", "emergency_signal", 0.75, True)

    monkeypatch.setattr(med, "EMERGENCY_TERMS", ("chest pain", "cannot breathe"))
    wrapped = _wrap_safety_check(original)
    decision = wrapped(object(), "I do not have chest pain but I cannot breathe")
    assert decision.emergency is True
