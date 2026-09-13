from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.evidence_guard import verify_claims
from rag_project.runtime_functionality_numeric_range_fix import _range_compatible, _wrap_numeric_verifier


def test_point_claim_is_supported_by_containing_evidence_range():
    assert _range_compatible("7 mg", "5-10 mg") is True


def test_unit_converted_point_claim_is_supported_by_containing_range():
    assert _range_compatible("7000 mcg", "5-10 mg") is True


def test_point_outside_evidence_range_is_rejected():
    assert _range_compatible("11 mg", "5-10 mg") is False


def test_answer_range_must_be_contained_by_evidence_range():
    assert _range_compatible("6-8 mg", "5-10 mg") is True
    assert _range_compatible("4-8 mg", "5-10 mg") is False


def test_numeric_verifier_accepts_point_from_evidence_range():
    hit = SimpleNamespace(text="The recommended dose is 5-10 mg.", metadata={})
    route = SimpleNamespace(numeric_sensitivity=True)
    original_result = {"numeric_mismatch": True, "grounding": {"allow": True}, "final_answer": {"allow": True}, "allow": False}
    wrapped = _wrap_numeric_verifier(lambda self, answer, hits, route, compiled: original_result)
    result = wrapped(SimpleNamespace(), "Use 7 mg.", [hit], route, {})
    assert result["numeric_mismatch"] is False
    assert result["allow"] is True


def test_claim_verification_accepts_point_inside_range_evidence():
    answer = "The dose is 7 mg. [S1]"
    evidence = ["The recommended dose is 5-10 mg."]
    checks = verify_claims(answer, evidence, ["S1"])
    assert len(checks) == 1
    assert checks[0].numeric_mismatch is False
