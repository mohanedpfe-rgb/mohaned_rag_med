from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class Hit:
    text: str
    score: float = 0.9
    metadata: dict | None = None
    doc_id: str = "doc-1"

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"document_id": self.doc_id, "chunk_id": "chunk-1", "page_numbers": [1]}


@pytest.mark.parametrize(
    ("claim", "evidence", "expected"),
    [
        ("Diabetes is chronic", "Diabetes is chronic.", "SUPPORTED"),
        ("Diabetes is not chronic", "Diabetes is chronic.", "CONTRADICTED"),
        ("The dose is 500 mg", "The recommended dose is 0.5 g.", "SUPPORTED"),
        ("The dose is 600 mg", "The recommended dose is 500 mg.", "NUMERIC_MISMATCH"),
        ("The moon is blue", "Diabetes is chronic.", "UNSUPPORTED"),
    ],
)
def test_claim_verification_adversarial_matrix(claim: str, evidence: str, expected: str) -> None:
    from rag_project.intelligence.evidence_guard import verify_claims

    checks = verify_claims(claim, [evidence], ["S1"])
    assert checks
    assert checks[0].status == expected


def test_numeric_units_are_cross_unit_compatible() -> None:
    from rag_project.intelligence.evidence_guard import numeric_consistency

    assert numeric_consistency("500 mg", "0.5 g")['mismatch'] is False
    assert numeric_consistency("1 L", "1000 ml")['mismatch'] is False
    assert numeric_consistency("1 min", "60 s")['mismatch'] is False
    assert numeric_consistency("1 kg", "1000 g")['mismatch'] is False


def test_numeric_mismatch_detects_incompatible_value() -> None:
    from rag_project.intelligence.evidence_guard import numeric_consistency

    result = numeric_consistency("500 mg", "0.6 g")
    assert result['checked'] is True
    assert result['mismatch'] is True
    assert result['unsupported_numeric']


def test_split_claims_handles_citations_and_bullets() -> None:
    from rag_project.intelligence.evidence_guard import split_claims

    claims = split_claims("- Diabetes is chronic. [S1]\n- Hypertension may coexist. [S2]")
    assert len(claims) == 2
    assert all(claims)


def test_citation_firewall_removes_blocked_content() -> None:
    from rag_project.intelligence.evidence_guard import ClaimCheck, citation_firewall

    checks = [
        ClaimCheck("Supported fact", 0.9, "SUPPORTED", ("S1",)),
        ClaimCheck("Unsafe unsupported fact", 0.0, "UNSUPPORTED", ()),
    ]
    safe, blocked = citation_firewall("original", checks)
    assert blocked is True
    assert "Supported fact" in safe
    assert "Unsafe unsupported fact" not in safe
    assert "withheld" in safe


def test_grounding_decision_thresholds() -> None:
    from rag_project.intelligence.evidence_guard import ClaimCheck, grounding_decision

    all_good = [ClaimCheck("a", .9, "SUPPORTED", ("S1",)), ClaimCheck("b", .8, "PARTIAL", ("S1",))]
    mixed = [ClaimCheck("a", .9, "SUPPORTED", ("S1",)), ClaimCheck("b", .1, "UNSUPPORTED", ())]
    assert grounding_decision(all_good)['allow'] is True
    assert grounding_decision(mixed)['allow'] is False
    assert grounding_decision([])['allow'] is False


def test_contradiction_report_counts_only_bad_claims() -> None:
    from rag_project.intelligence.evidence_guard import ClaimCheck, contradiction_report

    checks = [
        ClaimCheck("a", .9, "SUPPORTED", ("S1",)),
        ClaimCheck("b", .1, "CONTRADICTED", ("S2",), contradiction=True),
    ]
    report = contradiction_report(checks)
    assert report['has_contradiction'] is True
    assert report['count'] == 1


def test_claim_evidence_matrix_limits_selected_spans() -> None:
    from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix

    hits = [
        Hit("Diabetes is chronic and requires monitoring.", metadata={"document_id": "d1", "chunk_id": "c1", "page_numbers": [1]}),
        Hit("Diabetes is chronic.", metadata={"document_id": "d1", "chunk_id": "c2", "page_numbers": [2]}),
        Hit("The patient has diabetes.", metadata={"document_id": "d2", "chunk_id": "c3", "page_numbers": [3]}),
        Hit("Unrelated cardiology text.", metadata={"document_id": "d3", "chunk_id": "c4", "page_numbers": [4]}),
    ]
    matrix = build_claim_evidence_matrix(["Diabetes is chronic"], hits, ["S1", "S2", "S3", "S4"])
    assert len(matrix) == 1
    assert len(matrix[0].evidence) <= 3
    assert matrix[0].claim == "Diabetes is chronic"


def test_matrix_strong_support_requires_every_claim_entailed() -> None:
    from rag_project.intelligence.evidence_entailment import ClaimEvidenceRecord, matrix_has_strong_support

    strong = ClaimEvidenceRecord("a", "ENTAILED", .9, (), ())
    partial = ClaimEvidenceRecord("b", "PARTIALLY_ENTAILED", .6, (), ())
    assert matrix_has_strong_support([strong]) is True
    assert matrix_has_strong_support([strong, partial]) is False
    assert matrix_has_strong_support([]) is False


def test_final_answer_contract_allows_supported_answer() -> None:
    from rag_project.intelligence.final_answer_contract import verify_final_answer

    result = verify_final_answer("Diabetes is chronic.", [Hit("Diabetes is chronic.")], require_entailment=True)
    assert result['checked'] is True
    assert result['allow'] is True
    assert result['reason'] == 'verified'
    assert result['supported_ratio'] >= .6


def test_final_answer_contract_rejects_unsupported_answer() -> None:
    from rag_project.intelligence.final_answer_contract import verify_final_answer

    result = verify_final_answer("The moon is blue.", [Hit("Diabetes is chronic.")], require_entailment=False)
    assert result['allow'] is False
    assert result['reason'] in {'blocked_claims', 'support_ratio_below_threshold'}


def test_final_answer_contract_requires_entailment_for_hard_queries() -> None:
    from rag_project.intelligence.final_answer_contract import verify_final_answer

    result = verify_final_answer(
        "Diabetes is chronic.",
        [Hit("Diabetes is a chronic disease.")],
        require_entailment=True,
    )
    assert result['checked'] is True
    assert isinstance(result['matrix_all_entailed'], bool)


def test_entity_coverage_reports_missing_entities() -> None:
    from rag_project.intelligence.entity_coverage import score_entity_coverage

    evidence = [Hit("Diabetes is chronic.")]
    report = score_entity_coverage("What about diabetes and nephropathy?", evidence)
    assert report['entity_count'] >= 1
    assert report['coverage'] <= 1
    assert isinstance(report['missing'], list)


def test_entity_coverage_accepts_explicit_planned_entities() -> None:
    from rag_project.intelligence.entity_coverage import extract_query_entities, score_entity_coverage

    entities = extract_query_entities("What is this?", ["RareDrug-X"])
    assert "raredrug-x" in entities
    report = score_entity_coverage("What is this?", [Hit("RareDrug-X is studied.")], ["RareDrug-X"])
    assert report['entity_count'] >= 1
    assert report['coverage'] > 0


def test_entity_coverage_measurement_and_abbreviation_detection() -> None:
    from rag_project.intelligence.entity_coverage import extract_query_entities

    entities = extract_query_entities("Compare HbA1c 7% with HTA")
    joined = " ".join(entities)
    assert "hba1c" in joined
    assert any("7%" in item for item in entities)
    assert "hta" in joined


def test_top_level_phase_one_is_deterministic_without_llm() -> None:
    from rag_project.intelligence.top_level_pipeline import deterministic_phase1

    a = deterministic_phase1("What causes diabetic nephropathy?")
    b = deterministic_phase1("What causes diabetic nephropathy?")
    assert a == b
    assert a.intent in {"causal", "etiology", "other", "factual"}
    assert a.entities
    assert a.rewritten_queries


def test_top_level_rewrite_follow_up_is_contextual() -> None:
    from rag_project.intelligence.top_level_pipeline import rewrite_follow_up

    result = rewrite_follow_up(
        "What about this?",
        [("What is diabetes?", "Diabetes is a metabolic disease.")],
    )
    assert "What is diabetes?" in result
    assert "Follow-up:" not in result
    assert "diabetes" in result.casefold()


def test_top_level_medical_term_layer_extracts_units_and_abbreviations() -> None:
    from rag_project.intelligence.top_level_pipeline import medical_term_layer

    report = medical_term_layer("HTA 7% and metformin 500 mg")
    assert report['units']
    assert report['abbreviations']
    assert report['terms']


def test_precision_filter_deduplicates_sources() -> None:
    from rag_project.intelligence.top_level_pipeline import PhasePlan, precision_filter

    phase = PhasePlan(
        "factual", ("diabetes",), (), ("diabetes" ,), ("diabetes",), "low", False, False, False, False, "deterministic", .9
    )
    hits = [Hit("Diabetes is chronic.", .9, {"document_id":"d1","chunk_id":"c1"}), Hit("Diabetes is chronic.", .8, {"document_id":"d1","chunk_id":"c1"}), Hit("Unrelated text.", .1, {"document_id":"d2","chunk_id":"c2"})]
    selected = precision_filter(hits, phase, limit=8)
    keys = [(h.metadata['document_id'], h.metadata['chunk_id']) for h in selected]
    assert len(keys) == len(set(keys))


def test_compress_context_respects_character_budget() -> None:
    from rag_project.intelligence.top_level_pipeline import compress_context

    hits = [Hit("Diabetes is chronic. " * 100), Hit("Unrelated. " * 100)]
    text, stats = compress_context("diabetes", hits, max_chars=500)
    assert len(text) <= 500
    assert stats['selected_sentences'] >= 0
    assert 0 <= stats['compression_ratio'] <= 1


def test_dynamic_temperature_is_zero_for_high_risk_numeric_queries() -> None:
    from rag_project.intelligence.top_level_pipeline import PhasePlan, dynamic_temperature

    phase = PhasePlan("numeric", (), (), ("dose",), (), "low", False, True, False, False, "deterministic", .9)
    assert dynamic_temperature(phase, .8) == 0.0


def test_dynamic_temperature_allows_default_for_simple_queries() -> None:
    from rag_project.intelligence.top_level_pipeline import PhasePlan, dynamic_temperature

    phase = PhasePlan("factual", (), (), ("what is diabetes",), (), "low", False, False, False, False, "deterministic", .9)
    assert dynamic_temperature(phase, .2) == .2


def test_extractive_draft_marks_sentences_with_source_ids() -> None:
    from rag_project.intelligence.top_level_pipeline import PhasePlan, extractive_draft

    phase = PhasePlan("factual", ("diabetes",), (), ("diabetes",), (), "low", False, False, False, False, "deterministic", .9)
    draft, state = extractive_draft("diabetes", [Hit("Diabetes is chronic. The weather is unrelated.")], phase)
    assert "[S1]" in draft
    assert state['supported'] is True


def test_visibility_contract_reports_missing_fields_honestly() -> None:
    from rag_project.intelligence.ui_visibility_contract import build_visibility_contract

    incomplete = build_visibility_contract({})
    assert incomplete['signals_present'] is False
    assert incomplete['visible_signal_count'] < incomplete['required_signal_count']


def test_visibility_contract_reports_complete_payload() -> None:
    from rag_project.intelligence.ui_visibility_contract import build_visibility_contract

    payload = {
        "phase_plan": {"intent": "factual", "entities": ["diabetes"]},
        "rewritten_question": "What is diabetes?",
        "evidence_claim_matrix": [{"claim": "Diabetes is chronic", "status": "ENTAILED", "support": .9}],
        "confidence_calibration": {"calibrated": .91, "level": "high"},
        "abstention_reasons": [],
    }
    result = build_visibility_contract(payload)
    assert result['signals_present'] is True
    assert result['visible_signal_count'] == result['required_signal_count']
    assert result['intent'] == 'factual'
    assert result['entities'] == ['diabetes']
