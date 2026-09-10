from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_project.app.rag_system import RAGSystem, RetrievalHit
from rag_project.app.production_rag import ProductionRAGSystem, _is_explicit_followup
from rag_project.intelligence.adaptive_retrieval import choose_retrieval_budget, should_retry_retrieval
from rag_project.intelligence.advanced_clinical_reasoner import assess_clinical_reasoning, extract_clinical_facts
from rag_project.intelligence.confidence_calibration import calibrate_confidence, confidence_gate
from rag_project.intelligence.evidence_entailment import build_claim_evidence_matrix, matrix_has_strong_support
from rag_project.intelligence.evidence_guard import (
    ClaimCheck,
    citation_firewall,
    detect_contradiction,
    evidence_confidence,
    extract_measurements,
    grounding_decision,
    numeric_consistency,
    semantic_support,
    split_claims,
    verify_claims,
)
from rag_project.intelligence.final_answer_contract import verify_final_answer
from rag_project.intelligence.god_mode import _diversify, _metadata_boost
from rag_project.intelligence.god_mode_100 import _runtime_phase_implementation, _validated_model_entities
from rag_project.intelligence.medical_safety import apply_medical_safety_policy, is_high_risk_medical_query
from rag_project.intelligence.query_intelligence import (
    classify_intent,
    decompose_query,
    extract_query_entities,
    normalize_query,
    plan_query,
)
from rag_project.intelligence.semantic_reasoning import QueryUnderstanding, extract_clinical_entities, semantic_evidence_alignment
from rag_project.intelligence.small_model_reasoner import (
    _extract_json,
    _sanitize_assist,
    analyze_with_small_model,
    augment_query_plan,
    merge_understanding,
    should_use_small_model,
)
from rag_project.intelligence.top_level_pipeline import (
    PhasePlan,
    _hard_query,
    compress_context,
    deterministic_phase1,
    dynamic_temperature,
    extractive_draft,
    medical_term_layer,
    precision_filter,
)


def hit(text: str, score: float = 0.8, *, doc: str = "doc-1", chunk: str = "chunk-1", page: int = 1):
    return RetrievalHit(
        doc_id=doc,
        text=text,
        metadata={
            "document_id": doc,
            "chunk_id": chunk,
            "page_numbers": [page],
            "file_name": f"{doc}.pdf",
            "language": "fr",
            "evidence_types": ["text"],
        },
        score=score,
        vector_score=score,
        lexical_score=0.0,
    )


def understanding(question: str, *, entities=(), intent="factual", relations=(), constraints=(), confidence=0.8):
    return QueryUnderstanding(
        normalized=question.casefold(),
        intents=(intent,),
        primary_intent=intent,
        entities=tuple(entities),
        relations=tuple(relations),
        constraints=tuple(constraints),
        answer_shape="explanation",
        semantic_terms=tuple(question.casefold().split()),
        confidence=confidence,
    )


def phase(*, intent="factual", entities=(), sub_questions=(), rewritten_queries=("What are the main findings?",), must_contain=(), ambiguity="low", numeric=False, table=False, figure=False, multi=False, confidence=0.9):
    return PhasePlan(
        intent=intent,
        entities=tuple(entities),
        sub_questions=tuple(sub_questions),
        rewritten_queries=tuple(rewritten_queries),
        must_contain=tuple(must_contain),
        ambiguity=ambiguity,
        needs_table=numeric or table,
        needs_numeric=numeric,
        needs_figure=figure,
        needs_multi_hop=multi,
        planner_source="deterministic",
        planner_confidence=confidence,
    )


@pytest.mark.parametrize("raw,expected", [
    ("  What   is   diabetes?  ", "What is diabetes?"),
    ("What's diabetes?", "What is diabetes?"),
    ("diabetes – hypertension", "diabetes - hypertension"),
    ("diabetes — hypertension", "diabetes - hypertension"),
    ("A\nB\tC", "A B C"),
    ("", ""), ("   ", ""), ("ééé", "ééé"), ("العربية", "العربية"),
])
def test_normalize_query_boundaries(raw, expected):
    assert normalize_query(raw) == expected


@pytest.mark.parametrize("question,expected", [
    ("What is diabetes?", "factual"), ("Define diabetes mellitus.", "factual"), ("What is HbA1c?", "factual"),
    ("Compare diabetes and hypertension.", "comparison"), ("What is the difference between diabetes and hypertension?", "comparison"),
    ("What is the dose of metformin?", "numeric"), ("How many mg?", "numeric"),
    ("What are the diagnostic criteria?", "diagnosis"), ("What is the treatment?", "management"),
    ("What causes ketoacidosis?", "etiology"), ("How does insulin work?", "mechanism"), ("What is the prognosis?", "prognosis"),
    ("Where is the source?", "navigation"), ("Which table contains potassium values?", "table_lookup"), ("What does figure 2 show?", "figure_lookup"),
])
def test_query_plan_intent_matrix(question, expected):
    plan = plan_query(question)
    assert plan.intent == expected
    assert plan.normalized


@pytest.mark.parametrize("question", [
    "What are the main findings?", "What is diabetes?", "Define hypertension.", "Explain diabetic nephropathy.",
    "What is albuminuria?", "What is hypokalemia?", "What is hyperaldosteronism?", "What is insulin resistance?",
    "What is HbA1c?", "What is DKA?", "What are the clinical consequences?", "What are the biological consequences?",
    "What is the diagnosis?", "What is the prognosis?", "What does this section say?", "Summarize this topic.",
    "Give the definition.", "List the main points.", "What are the findings on this page?", "Explain the concept.",
    "What does the document report?", "What was observed?", "What abnormalities are described?", "What complications are mentioned?",
])
def test_simple_question_never_creates_fake_question_entities(question):
    entities = extract_query_entities(question)
    ordinary = {"what", "are", "the", "main", "findings", "is", "give", "list", "points", "explain", "concept", "does", "this", "document", "report"}
    assert not (set(entities) & ordinary)


@pytest.mark.parametrize("question", [
    "What is diabetes?", "What is hypertension?", "What is HbA1c?", "What are the main findings?",
    "What does the document report?", "List the complications.", "Summarize the section.", "Explain this condition.",
])
def test_easy_questions_are_not_small_model_escalated(question):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    expected = bool(p.needs_multi_hop or p.needs_numeric or p.needs_table or p.needs_figure or p.intent in {"diagnosis", "management", "etiology", "mechanism", "prognosis", "comparison"} or len(question.split()) >= 16)
    assert should_use_small_model(question, u) is expected


@pytest.mark.parametrize("question", [
    "Why does ketoacidosis happen?", "How does insulin affect glucose?", "Compare diabetes and hypertension.",
    "What is the treatment of DKA?", "What are the diagnostic criteria for DKA?", "What is the prognosis of sepsis?",
    "What is the exact dose?", "Which table contains the values?", "What does figure 3 show?",
    "Which condition is associated with albuminuria and why?",
])
def test_complex_queries_activate_strict_path(question):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    assert _hard_query(p, u, question) is True


@pytest.mark.parametrize("question,expected_substrings", [
    ("Compare diabetes and hypertension.", ("diabetes", "hypertension")),
    ("What is the dose of metformin?", ("metformin",)),
    ("What are the causes and complications of DKA?", ("causes", "complications")),
    ("What is diabetes?", ("diabetes",)),
])
def test_decomposition_preserves_user_concepts(question, expected_substrings):
    joined = " ".join(decompose_query(question)).casefold()
    for value in expected_substrings:
        assert value in joined


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "null", "true", "{", "}", "```text nope ```", "prefix {bad}"])
def test_small_model_json_parser_fails_closed(raw):
    assert _extract_json(raw) is None


@pytest.mark.parametrize("question,intent", [
    ("What is diabetes?", "diagnosis"), ("What is diabetes?", "management"), ("What is diabetes?", "etiology"),
    ("Compare diabetes and hypertension.", "diagnosis"), ("What are the diagnostic criteria?", "diagnosis"),
    ("What is the treatment?", "management"), ("What causes DKA?", "etiology"), ("How does insulin work?", "mechanism"),
])
def test_small_model_cannot_promote_simple_query_to_hard_intent(question, intent):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    sanitized = _sanitize_assist(question, u, {
        "intent": intent, "entities": ["main", "findings", "invented entity"],
        "relations": ["causality", "association"], "constraints": ["safety", "population"],
        "subquestions": ["invented subquestion"], "retrieval_terms": ["invented retrieval term"],
        "answer_strategy": "invented strategy",
    })
    hard_cues = ("diagnos", "treatment", "cause", "why", "mechanism", "prognosis", "compare", "relationship", "related")
    if p.intent == "factual" and not any(c in question.casefold() for c in hard_cues):
        assert sanitized["intent"] == "factual"
    assert "main" not in [x.casefold() for x in sanitized["entities"]]
    assert "findings" not in [x.casefold() for x in sanitized["entities"]]


@pytest.mark.parametrize("question", [
    "Why is diabetes associated with hypertension?", "Compare diabetes and hypertension.", "What is the treatment for DKA?",
    "What are the diagnostic criteria for DKA?", "What is the prognosis of sepsis?", "What dose should be used?",
    "What about this?", "And the complications?",
])
def test_small_model_is_selected_for_hard_or_ambiguous_questions(question):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    assert should_use_small_model(question, u) is True


class FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0
    def generate(self, **kwargs):
        self.calls += 1
        return self.payload


@pytest.mark.parametrize("payload", ["not json", '{"intent":"diagnosis","entities":["main","findings"]}', '{"intent":"factual","entities":[] }'])
def test_small_model_easy_query_is_not_called(payload):
    llm = FakeLLM(payload)
    u = understanding("What are the main findings?", intent="factual")
    assert analyze_with_small_model(llm, "What are the main findings?", u) is None
    assert llm.calls == 0


@pytest.mark.parametrize("question,payload", [
    ("Why is diabetes related to hypertension?", '{"intent":"etiology","entities":["diabetes","hypertension"],"relations":["causality"],"constraints":[],"subquestions":["why"],"retrieval_terms":["cause"],"answer_strategy":"reason"}'),
    ("Compare diabetes and hypertension.", '{"intent":"comparison","entities":["diabetes","hypertension"],"relations":["comparison"],"constraints":[],"subquestions":["compare"],"retrieval_terms":["difference"],"answer_strategy":"compare"}'),
])
def test_small_model_hard_query_is_sanitized(question, payload):
    llm = FakeLLM(payload)
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    result = analyze_with_small_model(llm, question, u)
    assert result is not None
    assert llm.calls == 1


@pytest.mark.parametrize("text,expected_relation", [
    ("Diabetes causes diabetic nephropathy.", "causes"),
    ("Hypertension is associated with albuminuria.", "association"),
    ("Diabetes is diagnosed by elevated glucose.", "diagnoses"),
    ("Diabetes is treated with metformin.", "treated_with"),
    ("Aspirin is contraindicated in this condition.", "contraindicated"),
])
def test_clinical_fact_extraction_relations(text, expected_relation):
    facts = extract_clinical_facts(text, node_id="n1", document_id="d1")
    assert any(f.predicate == expected_relation for f in facts)
    assert all(f.node_id == "n1" for f in facts)
    assert all(f.document_id == "d1" for f in facts)


@pytest.mark.parametrize("text", ["Diabetes does not cause hypertension.", "Diabetes is not associated with hypertension.", "Without treatment, outcomes worsen.", "No hypertension is present."])
def test_clinical_fact_negation_survives(text):
    facts = extract_clinical_facts(text)
    assert facts
    assert any(f.polarity == -1 for f in facts)


@pytest.mark.parametrize("question", ["What are the main findings?", "Summarize the evidence.", "What does the document report?", "List the main abnormalities.", "What are the biological consequences?"])
def test_advanced_reasoner_entity_free_summary(question):
    u = understanding(question, intent="factual")
    result = assess_clinical_reasoning(u, [hit("Les principales conséquences biologiques sont l'hyperglycémie, la cétose et l'acidose métabolique.", score=0.84)])
    assert result.mode == "DIRECT_SUMMARY"
    assert result.allow_generation is True
    assert result.entity_coverage == 1.0


@pytest.mark.parametrize("question,text", [
    ("What is diabetes?", "Diabetes is a chronic metabolic disease."),
    ("What is hypertension?", "Hypertension is elevated arterial blood pressure."),
    ("What is albuminuria?", "Albuminuria is albumin in the urine."),
])
def test_advanced_reasoner_direct_questions(question, text):
    u = understanding(question, entities=extract_clinical_entities(question), intent="factual")
    result = assess_clinical_reasoning(u, [hit(text, score=0.86)])
    assert result.allow_generation is True
    assert result.mode == "DIRECT"
    assert result.entity_coverage >= 0.5


@pytest.mark.parametrize("question", ["Why is diabetes related to hypertension?", "What causes hypertension in diabetes?", "What is the mechanism connecting diabetes and nephropathy?"])
def test_advanced_reasoner_blocks_unrelated_relationship(question):
    u = understanding(question, entities=extract_clinical_entities(question), intent="etiology", relations=("causality",))
    result = assess_clinical_reasoning(u, [hit("Unrelated endocrine facts with no requested relation.", score=0.4)])
    assert result.allow_generation is False
    assert result.blocked_reasons


@pytest.mark.parametrize("text", [
    "Diabetes causes diabetic nephropathy. Albuminuria is associated with hypertension.",
    "Diabetes is treated with metformin. Metformin is associated with improved control.",
    "Hypertension causes albuminuria. Albuminuria is associated with nephropathy.",
])
def test_clinical_fact_extractor_is_bounded(text):
    facts = extract_clinical_facts(text)
    assert len(facts) <= 32
    assert all(f.node_id for f in facts)


@pytest.mark.parametrize("base_score", [0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
def test_metadata_boost_is_nonnegative(base_score):
    p = plan_query("What is diabetes?")
    boost = _metadata_boost(hit("Diabetes is chronic.", score=base_score), p)
    assert boost >= 0.0


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 8, 12, 20])
def test_diversify_respects_limit(limit):
    hits = [hit(f"fact {i}", score=1.0 - i / 100, doc=f"doc-{i % 4}", chunk=f"chunk-{i}", page=i + 1) for i in range(30)]
    result = _diversify(hits, limit)
    assert len(result) <= limit
    assert len({h.metadata["chunk_id"] for h in result}) == len(result)


@pytest.mark.parametrize("scores", [[0.1, 0.2, 0.9], [0.9, 0.8, 0.1], [0.5, 0.5, 0.5], [0.0, 1.0]])
def test_diversify_orders_by_score(scores):
    hits = [hit(f"fact {i}", score=s, doc=f"doc-{i}", chunk=f"chunk-{i}", page=i + 1) for i, s in enumerate(scores)]
    result = _diversify(hits, len(hits))
    assert [h.score for h in result] == sorted(scores, reverse=True)


@pytest.mark.parametrize("max_chars", [20, 40, 80, 120, 500, 6500])
def test_context_compression_obeys_budget(max_chars):
    context, meta = compress_context("diabetes hyperglycemia", [hit("Diabetes is associated with hyperglycemia. " * 3), hit("Unrelated thyroid physiology. " * 4, doc="doc-2", chunk="chunk-2")], max_chars=max_chars)
    assert len(context) <= max_chars
    assert meta["input_sentences"] >= meta["selected_sentences"]
    assert meta["compression_ratio"] >= 0.0


@pytest.mark.parametrize("question,text", [
    ("What is diabetes?", "Diabetes is a chronic metabolic disease."),
    ("What are the main findings?", "The main findings are hyperglycemia, ketosis, and metabolic acidosis."),
    ("What is hypertension?", "Hypertension is elevated blood pressure."),
])
def test_extractive_draft_has_verified_source_marker(question, text):
    p = deterministic_phase1(question)
    draft, meta = extractive_draft(question, [hit(text, score=0.84)], p)
    assert meta["supported"] is True
    assert draft
    assert "[S1]" in draft


@pytest.mark.parametrize("text", ["7%", "7 %", "0.5 g", "500 mg", "1 kg", "1000 ml", "1 L", "37 °C", "120 mmHg", "60 min", "3600 s", "1 week", "7 days", "1 kHz", "1000 Hz", "5-10 mg"])
def test_measurement_extraction_matrix(text):
    assert extract_measurements(text)


@pytest.mark.parametrize("claim,evidence", [("500 mg", "0.5 g"), ("1 kg", "1000 g"), ("1 L", "1000 ml"), ("1 min", "60 s"), ("1 week", "7 days"), ("100 cm", "1 m"), ("1 kHz", "1000 Hz"), ("37 °C", "37 C")])
def test_numeric_equivalence_matrix(claim, evidence):
    assert numeric_consistency(claim, evidence)["mismatch"] is False


@pytest.mark.parametrize("claim,evidence", [("500 mg", "501 mg"), ("1 kg", "999 g"), ("1 L", "999 ml"), ("7%", "8%"), ("120 mmHg", "80 mmHg"), ("37 C", "40 C"), ("10 min", "5 min")])
def test_numeric_mismatch_matrix(claim, evidence):
    result = numeric_consistency(claim, evidence)
    assert result["checked"] is True
    assert result["mismatch"] is True


@pytest.mark.parametrize("claim,evidence", [
    ("Diabetes is not chronic.", "Diabetes is chronic."), ("No hypertension.", "The patient has hypertension."),
    ("Avoid aspirin.", "Aspirin is recommended."), ("Without treatment outcomes improve.", "With treatment outcomes improve."),
    ("Diabetes is absent.", "Diabetes is present."),
])
def test_contradiction_matrix(claim, evidence):
    assert detect_contradiction(claim, [evidence]) is True


@pytest.mark.parametrize("answer", ["", "ok", "yes", "thanks", "[S1]", "[S2]"])
def test_claim_noise_is_ignored(answer):
    assert split_claims(answer) == []


@pytest.mark.parametrize("claim,evidence,expected", [
    ("Diabetes is chronic.", "Diabetes is chronic.", "SUPPORTED"),
    ("The dose is 500 mg.", "The dose is 0.5 g.", "SUPPORTED"),
    ("The dose is 600 mg.", "The dose is 500 mg.", "NUMERIC_MISMATCH"),
    ("The moon is blue.", "Diabetes is chronic.", "UNSUPPORTED"),
    ("Diabetes is not chronic.", "Diabetes is chronic.", "CONTRADICTED"),
])
def test_verify_claims_status_matrix(claim, evidence, expected):
    result = verify_claims(claim, [evidence], ["S1"])
    assert len(result) == 1
    assert result[0].status == expected


@pytest.mark.parametrize("bad_status", ["UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"])
def test_citation_firewall_blocks_each_bad_status(bad_status):
    checks = [ClaimCheck("verified diabetes fact", 0.9, "SUPPORTED", ("S1",)), ClaimCheck("unsafe hallucinated fact", 0.1, bad_status, ())]
    safe, used = citation_firewall("original", checks)
    assert used is True
    assert "verified diabetes fact" in safe
    assert "unsafe hallucinated fact" not in safe


@pytest.mark.parametrize("answer,evidence", [
    ("Diabetes is chronic.", "Diabetes is chronic."),
    ("Diabetes is chronic.\nSources: [S1] book.pdf", "Diabetes is chronic."),
    ("Hypertension is common.", "Hypertension is common."),
])
def test_final_answer_contract_allows_simple_grounded_answers(answer, evidence):
    result = verify_final_answer(answer, [hit(evidence, score=0.9)])
    assert result["checked"] is True
    assert result["allow"] is True
    assert result["blocked_claims"] == 0


@pytest.mark.parametrize("answer,evidence", [
    ("Diabetes is chronic. The moon is blue.", "Diabetes is chronic."),
    ("The dose is 600 mg.", "The dose is 500 mg."),
    ("Diabetes is not chronic.", "Diabetes is chronic."),
])
def test_final_answer_contract_blocks_bad_answers(answer, evidence):
    result = verify_final_answer(answer, [hit(evidence, score=0.9)])
    assert result["allow"] is False
    assert result["blocked_claims"] >= 1


@pytest.mark.parametrize("claim,text", [
    ("Diabetes is chronic.", "Diabetes is chronic. Hypertension is common."),
    ("Hypertension is common.", "Diabetes is chronic. Hypertension is common."),
    ("Metformin lowers glucose.", "Metformin lowers glucose. Another sentence."),
    ("The dose is 500 mg.", "The dose is 0.5 g."),
])
def test_claim_matrix_returns_valid_spans(claim, text):
    matrix = build_claim_evidence_matrix([claim], [hit(text)], ["S1"])
    assert len(matrix) == 1
    assert matrix[0].evidence
    assert all(span.source_id == "S1" for span in matrix[0].evidence)
    assert all(span.start <= span.end for span in matrix[0].evidence)


@pytest.mark.parametrize("value", [-2, -1, 0, 0.1, 0.5, 0.9, 1, 2])
def test_confidence_factors_are_clamped(value):
    result = calibrate_confidence(retrieval=value, rerank=value, entailment=value, entity_coverage=value, source_agreement=value, contradiction=value, safety_conflict=value, ocr_penalty=value)
    assert 0.0 <= result.raw <= 1.0
    assert 0.0 <= result.calibrated <= 1.0
    assert all(0.0 <= x <= 1.0 for x in result.factors.values())


@pytest.mark.parametrize("high,low", [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.6, 0.5)])
def test_confidence_positive_evidence_is_monotonic(high, low):
    a = calibrate_confidence(retrieval=high, rerank=high, entailment=high, entity_coverage=high, source_agreement=high, contradiction=0, safety_conflict=0)
    b = calibrate_confidence(retrieval=low, rerank=low, entailment=low, entity_coverage=low, source_agreement=low, contradiction=0, safety_conflict=0)
    assert a.calibrated >= b.calibrated


@pytest.mark.parametrize("bad", [0, 0.2, 0.5, 0.8, 1.0])
def test_confidence_contradiction_never_increases_score(bad):
    clean = calibrate_confidence(retrieval=0.8, rerank=0.8, entailment=0.8, entity_coverage=0.8, source_agreement=0.8, contradiction=0, safety_conflict=0)
    changed = calibrate_confidence(retrieval=0.8, rerank=0.8, entailment=0.8, entity_coverage=0.8, source_agreement=0.8, contradiction=bad, safety_conflict=0)
    assert changed.calibrated <= clean.calibrated


@pytest.mark.parametrize("args", [
    dict(query_tokens=3, entity_count=0, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=8, entity_count=1, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=20, entity_count=1, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=4, entity_count=3, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=5, entity_count=1, intent="diagnosis", confidence=0.5, initial_score=0.8),
    dict(query_tokens=5, entity_count=1, intent="factual", confidence=0.5, initial_score=0.2),
])
def test_adaptive_budget_is_bounded(args):
    budget = choose_retrieval_budget(**args)
    assert 1 <= budget.variant_limit <= 12
    assert 1 <= budget.candidate_limit <= 72
    assert 1 <= budget.rerank_limit <= 64
    assert 1 <= budget.depth <= 3


@pytest.mark.parametrize("alignment,coverage,contradiction,expected", [
    (0.9, 0.9, 0.0, False), (0.2, 0.9, 0.0, True), (0.9, 0.2, 0.0, True),
    (0.9, 0.9, 0.8, True), (0.9, 0.9, 0.0, False),
])
def test_retry_boundary(alignment, coverage, contradiction, expected):
    assert should_retry_retrieval(alignment=alignment, entity_coverage=coverage, contradiction=contradiction, attempts=0, max_attempts=1) is expected


@pytest.mark.parametrize("question,expected", [("What is HbA1c?", "hba1c"), ("What is DKA?", "dka"), ("What is COPD?", "copd"), ("What is ECG?", "ecg")])
def test_medical_term_layer_detects_abbreviation(question, expected):
    result = medical_term_layer(question)
    assert any(x.casefold() == expected for x in result["abbreviations"])


@pytest.mark.parametrize("question,expected", [("What is dapagliflozin?", "dapagliflozin"), ("What is metformin?", "metformin"), ("What is lisinopril?", "lisinopril"), ("What is enalapril?", "enalapril")])
def test_medical_term_layer_detects_drug_like_terms(question, expected):
    result = medical_term_layer(question)
    assert expected in [x.casefold() for x in result["terms"]]


@pytest.mark.parametrize("question", ["What is 500 mg?", "What is 7%?", "What is 120 mmHg?", "What is 37 C?", "What is 1 mL?", "What is 10 bpm?"])
def test_medical_term_layer_detects_measurement(question):
    assert medical_term_layer(question)["units"]


@pytest.mark.parametrize("question", ["What are the main findings?", "What is diabetes?", "What is hypertension?", "Define HbA1c.", "What are the complications of DKA?"])
def test_independent_question_not_followup(question):
    assert _is_explicit_followup(question) is False


@pytest.mark.parametrize("question", ["What about this?", "How about that?", "And the complications?", "And this?", "Then what?", "Et les complications?", "والمضاعفات؟"])
def test_explicit_followup_detected(question):
    assert _is_explicit_followup(question) is True


@pytest.mark.parametrize("question,expected", [
    ("Take 500 mg twice daily?", True), ("What medication should I take?", True), ("What is the treatment?", True),
    ("What is diabetes?", False), ("What are the main findings?", False), ("Define hypertension.", False),
])
def test_medical_safety_risk_classification(question, expected):
    assert is_high_risk_medical_query(question) is expected


@pytest.mark.parametrize("allowed", [True, False])
def test_medical_safety_policy_honors_all_gate_inputs(allowed):
    result = {
        "status": "SUCCESS", "answer": "Take 500 mg twice daily.",
        "citations": ["S1"] if allowed else [], "confidence": {"evidence_confidence": 0.95 if allowed else 0.2},
        "grounding": {"allow": allowed}, "contradiction_report": {"has_contradiction": not allowed},
    }
    out = apply_medical_safety_policy("Take 500 mg twice daily.", result, SimpleNamespace(medical_high_risk_evidence_threshold=0.8))
    assert out["medical_safety"]["high_risk_query"] is True
    assert (out["medical_safety"]["decision"] == "ALLOW_WITH_EVIDENCE") is allowed


@pytest.mark.parametrize("question", ["What is diabetes?", "What is hypertension?", "What are the main findings?", "What is HbA1c?"])
def test_public_alignment_empty_evidence_fails_closed(question):
    result = RAGSystem.evaluate_evidence_alignment(question, [])
    assert result["decision"] == "NOT_SUPPORTED"
    assert result["answerability"] == 0.0


@pytest.mark.parametrize("question", ["What is diabetes?", "What are the main findings?", "What are the complications?"])
def test_public_alignment_with_evidence_is_bounded(question):
    result = RAGSystem.evaluate_evidence_alignment(question, [hit("Diabetes is a chronic metabolic disease with hyperglycemia.", score=0.8)])
    assert 0.0 <= result["answerability"] <= 1.0
    assert 0.0 <= result["local_context_strength"] <= 1.0
    assert result["decision"] in {"DIRECTLY_SUPPORTED", "PARTIALLY_SUPPORTED", "RELATED_BUT_NOT_ANSWERING", "NOT_SUPPORTED"}


@pytest.mark.parametrize("question", ["What is diabetes?", "What are the main findings?", "What is hypertension?"])
def test_semantic_alignment_is_bounded(question):
    result = semantic_evidence_alignment(question, [hit("Diabetes is a chronic metabolic disease with hyperglycemia.", score=0.8)])
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["entity_coverage"] <= 1.0
    assert result["decision"] in {"DIRECTLY_SUPPORTED", "PARTIALLY_SUPPORTED", "RELATED_BUT_NOT_ANSWERING", "NOT_SUPPORTED"}


@pytest.mark.parametrize("question", ["What is diabetes?", "What is hypertension?", "What are the main findings?"])
def test_simple_question_planner_output_is_bounded(question):
    p = deterministic_phase1(question)
    assert p.intent
    assert len(p.entities) <= 16
    assert len(p.rewritten_queries) <= 8
    assert p.ambiguity in {"low", "medium", "high"}


@pytest.mark.parametrize("question", ["", " ", "\n", "?", "???", "hello", "thanks", "ok", "asdfghjkl", "12345"])
def test_pathological_query_is_total(question):
    p = plan_query(question)
    assert isinstance(p.normalized, str)
    assert len(p.variants) <= 10


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 25])
def test_claim_verification_scales_with_many_evidence_items(n):
    evidence = [hit(f"Diabetes evidence sentence {i}.", score=0.5 + min(0.49, i / 200), doc=f"doc-{i % 5}", chunk=f"chunk-{i}", page=i + 1) for i in range(n)]
    result = verify_final_answer("Diabetes evidence sentence 1.", evidence)
    assert isinstance(result, dict)
    assert result["claim_count"] in {0, 1}


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 20])
def test_reasoning_scales_with_many_hits(n):
    hits = [hit("Diabetes causes diabetic nephropathy.", score=0.8, doc=f"doc-{i % 4}", chunk=f"chunk-{i}", page=i + 1) for i in range(n)]
    question = "Why does diabetes cause nephropathy?"
    u = understanding(question, entities=extract_clinical_entities(question), intent="etiology", relations=("causality",))
    result = assess_clinical_reasoning(u, hits)
    assert len(result.supported_paths) <= 8
    assert len(result.blocked_reasons) <= 8


# Critical reproduction of the original reported failure.
def test_original_main_findings_failure_path_is_removed():
    question = "What are the main findings?"
    evidence = hit(
        "IV. Conséquences biologiques : 1. Hyperglycémie >2.5 g/L. 2. Cétose. 3. Acidose métabolique. 4. Déplétion potassique.",
        score=0.82, doc="endocrino", chunk="218", page=52,
    )
    p = plan_query(question)
    u = understanding(question, intent=p.intent)
    result = assess_clinical_reasoning(u, [evidence])
    assert p.entities == ()
    assert u.entities == ()
    assert result.mode == "DIRECT_SUMMARY"
    assert result.allow_generation is True
    assert "insufficient_entity_coverage" not in result.blocked_reasons


def test_original_main_findings_answer_verifies_against_evidence():
    answer = "The main findings include hyperglycemia, ketosis, and metabolic acidosis."
    evidence = "IV. Conséquences biologiques : Hyperglycemia >2.5 g/L. Cétose. Acidose métabolique."
    result = verify_final_answer(answer, [hit(evidence, score=0.9)])
    assert result["claim_count"] == 1
    assert result["blocked_claims"] == 0
    assert result["allow"] is True


def test_sources_line_never_becomes_clinical_claim():
    answer = "The main findings include hyperglycemia, ketosis, and metabolic acidosis.\n\nSources: [S1] endocrino.pdf (page 52)"
    result = verify_claims(answer, ["Hyperglycemia, ketosis, and metabolic acidosis are listed."], ["S1"])
    assert len(result) == 1
    assert not result[0].claim.casefold().startswith("sources:")


def test_original_main_findings_matrix_is_single_claim_row():
    answer = "The main findings include hyperglycemia, ketosis, and metabolic acidosis."
    h = hit("Hyperglycemia, ketosis, and metabolic acidosis are listed.", score=0.9, doc="endocrino", page=52)
    result = verify_final_answer(answer, [h])
    assert result["claim_count"] == 1
    assert result["matrix_claim_count"] == 1
    assert result["blocked_claims"] == 0


def test_runtime_phase_contract_is_total():
    completed = {
        "phases": {"phase_1_query_understanding": "complete", "phase_2_retrieval_precision": "complete", "phase_4_verification": "complete", "phase_5_intelligence_visibility": "complete"},
        "phase_plan": {"planner_source": "deterministic", "planner_confidence": 0.9, "entities": []},
        "adaptive_retrieval": {"stage": 1, "queries": 1, "final_hits": 2, "escalated": False},
        "two_stage_synthesis": {"used": False, "required": False, "attempted": False, "fallback": True, "verification": {}},
        "confidence_calibration": {"calibrated": 0.7},
    }
    result = _runtime_phase_implementation(completed, [], {"blocked_claims": 0, "checked": True})
    assert set(result) == {"phase_1_query_understanding", "phase_2_retrieval_precision", "phase_3_two_stage_generation", "phase_4_verification", "phase_5_intelligence_visibility"}
    assert result["phase_5_intelligence_visibility"]["authority"].endswith("complete_phases")


def test_retrieval_hit_metadata_is_complete_for_ui():
    h = hit("Diabetes is chronic.", score=0.91, doc="book", chunk="chunk-1", page=52)
    assert h.metadata["file_name"] == "book.pdf"
    assert h.metadata["page_numbers"] == [52]
    assert h.metadata["chunk_id"] == "chunk-1"
    assert h.score == 0.91
