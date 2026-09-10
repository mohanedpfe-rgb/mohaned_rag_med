from __future__ import annotations

import math

import pytest

from rag_project.app.rag_system import EvidenceAlignment, QueryQualityClassifier
from rag_project.intelligence.evidence_guard import (
    ClaimCheck,
    citation_firewall,
    contradiction_report,
    evidence_confidence,
    grounding_decision,
    numeric_consistency,
    semantic_support,
    split_claims,
    verify_claims,
)
from rag_project.intelligence.final_44 import validate_citations


def test_claim_splitter_handles_sentences_and_bullets():
    claims = split_claims("- Dose is 500 mg.\n- Duration is 5 days.")
    assert claims == ["Dose is 500 mg.", "Duration is 5 days."]


def test_claim_splitter_ignores_tiny_fragments():
    assert split_claims("yes no") == []


def test_semantic_support_is_high_for_near_exact_evidence():
    score = semantic_support(
        "Diabetic nephropathy is a chronic microvascular complication of diabetes.",
        "La néphropathie diabétique est une complication microvasculaire chronique du diabète.",
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.05


def test_semantic_support_is_low_for_unrelated_medical_topics():
    score = semantic_support(
        "What are the diagnostic criteria for diabetic ketoacidosis?",
        "Papillary thyroid cancers are often associated with RET mutations.",
    )
    assert score < 0.35


@pytest.mark.parametrize(
    "claim,evidence",
    [
        ("The patient has no fever.", "The patient has fever."),
        ("Treatment is contraindicated.", "Treatment is indicated."),
        ("No potassium loss occurs.", "Potassium loss occurs."),
    ],
)
def test_negation_is_detected_as_a_possible_contradiction(claim, evidence):
    checks = verify_claims(claim, [evidence], ["S1"])
    assert checks
    assert checks[0].status == "CONTRADICTED" or checks[0].contradiction


def test_numeric_consistency_accepts_equivalent_units():
    result = numeric_consistency("Give 1000 mg.", "The dose is 1 g.")
    assert result["mismatch"] is False


def test_numeric_consistency_rejects_wrong_value_same_unit():
    result = numeric_consistency("Give 500 mg.", "The dose is 250 mg.")
    assert result["mismatch"] is True
    assert "500 mg" in result["unsupported_numeric"]


def test_numeric_consistency_rejects_wrong_unit_even_when_number_matches():
    result = numeric_consistency("Give 5 mg.", "The dose is 5 mL.")
    assert result["mismatch"] is True


def test_numeric_consistency_handles_ranges():
    result = numeric_consistency("Target is 2-4 mg.", "Target is 2-4 mg.")
    assert result["mismatch"] is False


def test_verify_claims_marks_direct_claim_supported():
    evidence = "Metformin is recommended as first-line treatment for type 2 diabetes in this document."
    claims = verify_claims("Metformin is recommended as first-line treatment for type 2 diabetes. [S1]", [evidence], ["S1"])
    assert claims
    assert claims[0].status in {"SUPPORTED", "PARTIAL"}
    assert claims[0].support > 0


def test_verify_claims_marks_unrelated_claim_unsupported():
    claims = verify_claims("The moon is made of cheese. [S1]", ["Insulin lowers blood glucose."], ["S1"])
    assert claims
    assert claims[0].status in {"UNSUPPORTED", "WEAK"}


def test_grounding_allows_fully_supported_claims():
    claims = [
        ClaimCheck("A", 0.9, "SUPPORTED", ("S1",)),
        ClaimCheck("B", 0.8, "PARTIAL", ("S2",)),
    ]
    decision = grounding_decision(claims, min_supported_ratio=0.60)
    assert decision["allow"] is True
    assert decision["blocked_claims"] == 0


def test_grounding_blocks_one_unsafe_claim_even_when_ratio_is_high():
    claims = [
        ClaimCheck("A", 0.9, "SUPPORTED", ("S1",)),
        ClaimCheck("B", 0.9, "SUPPORTED", ("S2",)),
        ClaimCheck("C", 0.1, "UNSUPPORTED", ()),
    ]
    decision = grounding_decision(claims, min_supported_ratio=0.60)
    assert decision["allow"] is False
    assert decision["blocked_claims"] == 1


def test_grounding_empty_claim_set_fails_closed():
    decision = grounding_decision([])
    assert decision["allow"] is False
    assert decision["supported_ratio"] == 0.0


def test_citation_firewall_keeps_supported_claims_and_withholds_unsafe_claims():
    claims = [
        ClaimCheck("Dose is 500 mg", 0.9, "SUPPORTED", ("S1",)),
        ClaimCheck("Dose is 900 mg", 0.1, "NUMERIC_MISMATCH", ("S2",), numeric_mismatch=True),
    ]
    text, used = citation_firewall("Dose is 500 mg. Dose is 900 mg.", claims)
    assert used is True
    assert "Dose is 500 mg" in text
    assert "900 mg" not in text


def test_citation_firewall_does_not_modify_fully_safe_answer():
    claims = [ClaimCheck("Dose is 500 mg", 0.9, "SUPPORTED", ("S1",))]
    text, used = citation_firewall("Dose is 500 mg.", claims)
    assert text == "Dose is 500 mg."
    assert used is False


def test_contradiction_report_is_false_when_no_claim_is_contradicted():
    claims = [ClaimCheck("A", 0.9, "SUPPORTED", ("S1",))]
    report = contradiction_report(claims)
    assert report["has_contradiction"] is False
    assert report["count"] == 0


def test_contradiction_report_counts_contradicted_claims():
    claims = [ClaimCheck("A", 0.2, "CONTRADICTED", ("S1",), contradiction=True)]
    report = contradiction_report(claims)
    assert report["has_contradiction"] is True
    assert report["count"] == 1


def test_evidence_confidence_is_bounded():
    for values in [
        dict(retrieval=0, rerank=0, entailment=0, quality=0),
        dict(retrieval=1, rerank=1, entailment=1, quality=1),
        dict(retrieval=100, rerank=100, entailment=100, quality=100),
        dict(retrieval=-1, rerank=-1, entailment=-1, quality=-1),
    ]:
        score = evidence_confidence(**values)
        assert 0.0 <= score <= 1.0
        assert math.isfinite(score)


def test_french_question_alignment_has_nonzero_signal():
    hits = [
        type("Hit", (), {"text": "La néphropathie diabétique est une complication microvasculaire du diabète."})()
    ]
    result = EvidenceAlignment.evaluate("Qu'est-ce que la néphropathie diabétique ?", hits)
    assert result["answerability"] > 0.0
    assert result["local_context_strength"] > 0.0


def test_english_question_with_french_evidence_is_not_silently_zeroed():
    hits = [
        type("Hit", (), {"text": "La néphropathie diabétique est une complication chronique du diabète avec albuminurie."})()
    ]
    result = EvidenceAlignment.evaluate("What is diabetic nephropathy and albuminuria?", hits)
    assert result["answerability"] > 0.0


def test_arabic_question_path_does_not_crash():
    result = QueryQualityClassifier.assess("ما هي مضاعفات داء السكري؟")
    assert isinstance(result, dict)
    assert "query_quality" in result


def test_validate_citations_turns_out_of_range_source_into_placeholder():
    answer, state = validate_citations("Fact [S1]. Bad [S4].", [object(), object()])
    assert "[S1]" in answer
    assert "[S?]" in answer
    assert state["invalid"] == [4]


def test_validate_citations_accepts_multiple_valid_sources():
    answer, state = validate_citations("Fact [S1] and fact [S2].", [object(), object()])
    assert answer.endswith(".")
    assert state["valid"] == [1, 2]
    assert state["invalid"] == []


def test_evidence_alignment_does_not_reward_more_irrelevant_hits():
    one = [type("Hit", (), {"text": "Thyroid cancer is often papillary."})()]
    many = one * 12
    r1 = EvidenceAlignment.evaluate("What is the treatment of diabetic ketoacidosis?", one)
    r2 = EvidenceAlignment.evaluate("What is the treatment of diabetic ketoacidosis?", many)
    assert r2["answerability"] <= r1["answerability"] + 0.01


def test_evidence_confidence_penalizes_contradiction():
    base = evidence_confidence(retrieval=0.8, rerank=0.8, entailment=0.8, quality=0.8, contradiction=0.0)
    bad = evidence_confidence(retrieval=0.8, rerank=0.8, entailment=0.8, quality=0.8, contradiction=1.0)
    assert bad < base
