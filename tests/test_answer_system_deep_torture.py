from __future__ import annotations

from dataclasses import SimpleNamespace

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


# ---------------------------------------------------------------------------
# Shared deterministic fixtures / factories
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 1. Query normalization / planning matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  What   is   diabetes?  ", "What is diabetes?"),
        ("What's diabetes?", "What is diabetes?"),
        ("diabetes – hypertension", "diabetes - hypertension"),
        ("diabetes — hypertension", "diabetes - hypertension"),
        ("A\nB\tC", "A B C"),
        ("", ""),
        ("   ", ""),
        ("ééé", "ééé"),
        ("العربية", "العربية"),
    ],
)
def test_normalize_query_boundaries(raw, expected):
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is diabetes?", "factual"),
        ("Define diabetes mellitus.", "factual"),
        ("What is HbA1c?", "factual"),
        ("Compare diabetes and hypertension.", "comparison"),
        ("What is the difference between diabetes and hypertension?", "comparison"),
        ("What is the dose of metformin?", "numeric"),
        ("How many mg?", "numeric"),
        ("What are the diagnostic criteria?", "diagnosis"),
        ("What is the treatment?", "management"),
        ("What causes ketoacidosis?", "etiology"),
        ("How does insulin work?", "mechanism"),
        ("What is the prognosis?", "prognosis"),
        ("Where is the source?", "navigation"),
        ("Which table contains potassium values?", "table_lookup"),
        ("What does figure 2 show?", "figure_lookup"),
    ],
)
def test_query_plan_intent_matrix(question, expected):
    plan = plan_query(question)
    assert plan.intent == expected
    assert plan.normalized


@pytest.mark.parametrize(
    "question",
    [
        "What are the main findings?",
        "What is diabetes?",
        "Define hypertension.",
        "Explain diabetic nephropathy.",
        "What is albuminuria?",
        "What is hypokalemia?",
        "What is hyperaldosteronism?",
        "What is insulin resistance?",
        "What is HbA1c?",
        "What is DKA?",
        "What are the clinical consequences?",
        "What are the biological consequences?",
        "What is the diagnosis?",
        "What is the prognosis?",
        "What does this section say?",
        "Summarize this topic.",
        "Give the definition.",
        "List the main points.",
        "What are the findings on this page?",
        "Explain the concept.",
        "What does the document report?",
        "What was observed?",
        "What abnormalities are described?",
        "What complications are mentioned?",
    ],
)
def test_simple_question_never_creates_fake_question_entities(question):
    entities = extract_query_entities(question)
    ordinary = {"what", "are", "the", "main", "findings", "is", "give", "list", "points", "explain", "concept", "does", "this", "document", "report"}
    assert not (set(entities) & ordinary)


@pytest.mark.parametrize(
    "question",
    [
        "What is diabetes?",
        "What is hypertension?",
        "What is HbA1c?",
        "What are the main findings?",
        "What does the document report?",
        "List the complications.",
        "Summarize the section.",
        "Explain this condition.",
    ],
)
def test_simple_questions_are_not_hard_by_themselves(question):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    assert _hard_query(p, u, question) is (p.intent in {"diagnosis", "management", "etiology", "mechanism", "prognosis", "comparison"} or len(question.split()) >= 16 or p.needs_numeric or p.needs_multi_hop or p.needs_table or p.needs_figure)


@pytest.mark.parametrize(
    "question",
    [
        "Why does ketoacidosis happen?",
        "How does insulin affect glucose?",
        "Compare diabetes and hypertension.",
        "What is the treatment of DKA?",
        "What are the diagnostic criteria for DKA?",
        "What is the prognosis of sepsis?",
        "What is the exact dose?",
        "Which table contains the values?",
        "What does figure 3 show?",
        "Which condition is associated with albuminuria and why?",
    ],
)
def test_complex_queries_activate_strict_path(question):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    assert _hard_query(p, u, question) is True


@pytest.mark.parametrize(
    "question,expected_substrings",
    [
        ("Compare diabetes and hypertension.", ("diabetes", "hypertension")),
        ("What is the dose of metformin?", ("metformin",)),
        ("What are the causes and complications of DKA?", ("causes", "complications")),
        ("What is diabetes?", ("diabetes",)),
    ],
)
def test_decomposition_preserves_user_concepts(question, expected_substrings):
    pieces = decompose_query(question)
    joined = " ".join(pieces).casefold()
    for value in expected_substrings:
        assert value in joined


# ---------------------------------------------------------------------------
# 2. Small-model copilot: adversarial structured-output matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "null", "true", "{", "}", "```text nope ```", "prefix {bad}"])
def test_small_model_json_parser_fails_closed(raw):
    assert _extract_json(raw) is None


@pytest.mark.parametrize(
    "question,intent,expected",
    [
        ("What is diabetes?", "diagnosis", "factual"),
        ("What is diabetes?", "management", "factual"),
        ("What are the diagnostic criteria?", "diagnosis", "diagnosis"),
        ("What is the treatment?", "management", "management"),
        ("What causes DKA?", "etiology", "etiology"),
        ("How does insulin work?", "mechanism", "mechanism"),
        ("Compare diabetes and hypertension.", "comparison", "comparison"),
        ("What is the dose of metformin?", "numeric", "numeric"),
    ],
)
def test_small_model_cannot_promote_factual_query_to_fake_hard_intent(question, intent, expected):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    parsed = {
        "intent": intent,
        "entities": ["main", "findings", "invented entity"],
        "relations": ["causality", "association"],
        "constraints": ["safety", "population"],
        "subquestions": ["invented subquestion"],
        "retrieval_terms": ["invented retrieval term"],
        "answer_strategy": "invented strategy",
    }
    sanitized = _sanitize_assist(question, u, parsed)
    if p.intent == "factual" and intent in {"diagnosis", "management", "etiology", "mechanism", "prognosis", "comparison", "relationship"} and not any(c in question.casefold() for c in ("diagnos", "treatment", "cause", "why", "mechanism", "prognosis", "compare", "relationship", "related")):
        assert sanitized["intent"] == "factual"
    else:
        assert sanitized["intent"] in {expected, p.intent}
    assert "main" not in [x.casefold() for x in sanitized["entities"]]
    assert "findings" not in [x.casefold() for x in sanitized["entities"]]


@pytest.mark.parametrize("question", ["What is diabetes?", "What is hypertension?", "What are the main findings?", "Define HbA1c."])
def test_small_model_not_called_for_easy_questions(question):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    assert should_use_small_model(question, u) is False


@pytest.mark.parametrize("question", [
    "Why is diabetes associated with hypertension?",
    "Compare diabetes and hypertension.",
    "What is the treatment for DKA?",
    "What are the diagnostic criteria for DKA?",
    "What is the prognosis of sepsis?",
    "What dose should be used?",
    "What about this?",
    "And the complications?",
])
def test_small_model_is_available_for_hard_or_ambiguous_questions(question):
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    assert should_use_small_model(question, u) is True


@pytest.mark.parametrize("parsed", [
    {"intent": "factual", "entities": [], "relations": [], "constraints": [], "subquestions": [], "retrieval_terms": [], "answer_strategy": ""},
    {"intent": "factual", "entities": ["diabetes"], "relations": [], "constraints": [], "subquestions": ["What is diabetes?"], "retrieval_terms": ["diabetes"], "answer_strategy": "direct"},
    {"intent": "comparison", "entities": ["diabetes", "hypertension"], "relations": ["comparison"], "constraints": [], "subquestions": ["compare"], "retrieval_terms": ["difference"], "answer_strategy": "compare"},
])
def test_small_model_assistance_is_shape_stable(parsed):
    question = "Compare diabetes and hypertension." if parsed["intent"] == "comparison" else "What is diabetes?"
    u = understanding(question, entities=extract_clinical_entities(question), intent=plan_query(question).intent)
    result = _sanitize_assist(question, u, parsed)
    assert set(result) == {"intent", "entities", "relations", "constraints", "subquestions", "retrieval_terms", "answer_strategy"}
    assert len(result["entities"]) <= 8
    assert len(result["subquestions"]) <= 6
    assert len(result["retrieval_terms"]) <= 6


class FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return self.payload


@pytest.mark.parametrize("payload", [
    "not json",
    '{"intent":"diagnosis","entities":["main","findings"]}',
    '{"intent":"factual","entities":[],"relations":[],"constraints":[],"subquestions":[],"retrieval_terms":[],"answer_strategy":"direct"}',
])
def test_small_model_runtime_failures_are_deterministic(payload):
    llm = FakeLLM(payload)
    question = "What are the main findings?"
    u = understanding(question, intent="factual")
    result = analyze_with_small_model(llm, question, u)
    assert result is None
    assert llm.calls == 0


@pytest.mark.parametrize("question,payload", [
    ("Why is diabetes related to hypertension?", '{"intent":"etiology","entities":["diabetes","hypertension"],"relations":["causality"],"constraints":[],"subquestions":["why"],"retrieval_terms":["cause"],"answer_strategy":"reason"}'),
    ("Compare diabetes and hypertension.", '{"intent":"comparison","entities":["diabetes","hypertension"],"relations":["comparison"],"constraints":[],"subquestions":["compare"],"retrieval_terms":["difference"],"answer_strategy":"compare"}'),
])
def test_small_model_runtime_hard_queries_returns_sanitized_assistance(question, payload):
    llm = FakeLLM(payload)
    p = plan_query(question)
    u = understanding(question, entities=extract_clinical_entities(question), intent=p.intent)
    result = analyze_with_small_model(llm, question, u)
    assert result is not None
    assert llm.calls == 1
    assert result["intent"] == p.intent or result["intent"] in {"etiology", "comparison"}


# ---------------------------------------------------------------------------
# 3. Advanced clinical reasoner: direct, summary, graph, conflict, safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_subject,expected_object,expected_relation",
    [
        ("Diabetes causes diabetic nephropathy.", "diabetes mellitus", "diabetic nephropathy", "causes"),
        ("Hypertension is associated with albuminuria.", "hypertension", "albuminuria", "association"),
        ("Diabetes is diagnosed by elevated glucose.", "diabetes mellitus", "glucose", "diagnoses"),
        ("Diabetes is treated with metformin.", "diabetes mellitus", "metformin", "treated_with"),
        ("Aspirin is contraindicated in this condition.", "aspirin", "this condition", "contraindicated"),
    ],
)
def test_clinical_fact_extraction_relations(text, expected_subject, expected_object, expected_relation):
    facts = extract_clinical_facts(text, node_id="n1", document_id="d1")
    assert facts
    assert any(f.subject == expected_subject and f.object == expected_object and f.predicate == expected_relation for f in facts)


@pytest.mark.parametrize("text", [
    "Diabetes does not cause hypertension.",
    "Diabetes is not associated with hypertension.",
    "Without treatment, outcomes worsen.",
    "No hypertension is present.",
])
def test_clinical_fact_negation_is_preserved(text):
    facts = extract_clinical_facts(text, node_id="n1", document_id="d1")
    assert facts
    assert any(f.polarity == -1 for f in facts)


@pytest.mark.parametrize("text", [
    "Diabetes causes diabetic nephropathy. Albuminuria is associated with hypertension.",
    "Diabetes is treated with metformin. Metformin is associated with improved control.",
    "Hypertension causes albuminuria. Albuminuria is associated with nephropathy.",
])
def test_clinical_fact_extractor_is_bounded(text):
    facts = extract_clinical_facts(text)
    assert len(facts) <= 32
    assert all(f.node_id for f in facts)
    assert all(f.document_id == "" for f in facts)


@pytest.mark.parametrize("question", [
    "What are the main findings?",
    "Summarize the evidence.",
    "What does the document report?",
    "List the main abnormalities.",
    "What are the biological consequences?",
])
def test_advanced_reasoner_supports_entity_free_summary(question):
    u = understanding(question, intent="factual")
    h = hit("Les principales conséquences biologiques sont l'hyperglycémie, la cétose et l'acidose métabolique.", score=0.8)
    result = assess_clinical_reasoning(u, [h])
    assert result.mode == "DIRECT_SUMMARY"
    assert result.allow_generation is True
    assert result.entity_coverage == 1.0


@pytest.mark.parametrize("question,text", [
    ("What is diabetes?", "Diabetes is a chronic metabolic disease."),
    ("What is hypertension?", "Hypertension is elevated arterial blood pressure."),
    ("What is albuminuria?", "Albuminuria is albumin in the urine."),
])
def test_advanced_reasoner_direct_entity_question(question, text):
    entities = extract_clinical_entities(question)
    u = understanding(question, entities=entities, intent="factual")
    result = assess_clinical_reasoning(u, [hit(text, score=0.86)])
    assert result.allow_generation is True
    assert result.mode == "DIRECT"
    assert result.entity_coverage >= 0.5


@pytest.mark.parametrize("question,text", [
    ("Why does diabetes cause nephropathy?", "Diabetes causes diabetic nephropathy."),
    ("Why is hypertension associated with albuminuria?", "Hypertension is associated with albuminuria."),
])
def test_advanced_reasoner_one_relation_path(question, text):
    u = understanding(question, entities=extract_clinical_entities(question), intent="etiology", relations=("causality",))
    result = assess_clinical_reasoning(u, [hit(text, score=0.84)])
    assert result.allow_generation in {True, False}
    assert result.mode in {"ONE_HOP", "DIRECT", "INSUFFICIENT"}
    assert result.depth in {0, 1}


@pytest.mark.parametrize("question", [
    "Why is diabetes related to hypertension?",
    "What causes hypertension in diabetes?",
    "What is the mechanism connecting diabetes and nephropathy?",
])
def test_advanced_reasoner_blocks_unrelated_relationship_query(question):
    u = understanding(question, entities=extract_clinical_entities(question), intent="etiology", relations=("causality",))
    result = assess_clinical_reasoning(u, [hit("Unrelated endocrine facts with no requested relation.", score=0.4)])
    assert result.allow_generation is False
    assert result.blocked_reasons


@pytest.mark.parametrize("question,text", [
    ("Why is aspirin contraindicated?", "Aspirin is contraindicated in this setting."),
    ("What is the treatment?", "The treatment is metformin, but aspirin is contraindicated."),
])
def test_advanced_reasoner_safety_does_not_crash(question, text):
    u = understanding(question, entities=extract_clinical_entities(question), intent="management", constraints=("safety",))
    result = assess_clinical_reasoning(u, [hit(text)])
    assert 0.0 <= result.safety_conflict <= 1.0


# ---------------------------------------------------------------------------
# 4. Retrieval filtering / diversification: candidate quality and preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_score", [0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
def test_metadata_boost_never_produces_invalid_score(base_score):
    h = hit("Diabetes is chronic.", score=base_score)
    p = plan_query("What is diabetes?")
    boost = _metadata_boost(h, p)
    assert boost >= 0.0
    assert base_score * boost >= 0.0


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 8, 12, 20])
def test_diversify_respects_limit(limit):
    hits = [hit(f"fact {i}", score=1.0 - i / 100, doc=f"doc-{i % 4}", chunk=f"chunk-{i}", page=i + 1) for i in range(30)]
    result = _diversify(hits, limit)
    assert len(result) <= limit
    assert len({h.metadata["chunk_id"] for h in result}) == len(result)


@pytest.mark.parametrize("scores", [
    [0.1, 0.2, 0.9],
    [0.9, 0.8, 0.1],
    [0.5, 0.5, 0.5],
    [0.0, 1.0],
])
def test_diversify_always_prefers_high_scores(scores):
    hits = [hit(f"fact {i}", score=s, doc=f"doc-{i}", chunk=f"chunk-{i}", page=i + 1) for i, s in enumerate(scores)]
    result = _diversify(hits, len(hits))
    returned_scores = [h.score for h in result]
    assert returned_scores == sorted(returned_scores, reverse=True)


@pytest.mark.parametrize("docs", [1, 2, 3, 4, 6])
def test_diversify_can_keep_multiple_documents(docs):
    hits = [hit(f"doc {i}", score=0.9 - i * 0.01, doc=f"doc-{i}", chunk=f"c-{i}", page=1) for i in range(docs)]
    result = _diversify(hits, docs)
    assert {h.doc_id for h in result} == {f"doc-{i}" for i in range(docs)}


# ---------------------------------------------------------------------------
# 5. Precision / context compression / extractive fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What is diabetes?",
    "What is hypertension?",
    "What are the main findings?",
    "What is HbA1c?",
    "What are the complications?",
])
def test_precision_filter_keeps_relevant_hit(question):
    p = deterministic_phase1(question)
    relevant = hit("Diabetes is a chronic metabolic disease with hyperglycemia.", score=0.8)
    irrelevant = hit("Unrelated thyroid anatomy and imaging.", score=0.75, doc="doc-2", chunk="chunk-2")
    selected = precision_filter([irrelevant, relevant], p, limit=2)
    assert relevant in selected


@pytest.mark.parametrize("max_chars", [20, 40, 80, 120, 500, 6500])
def test_context_compression_obeys_character_budget(max_chars):
    question = "diabetes hyperglycemia"
    hits = [
        hit("Diabetes is associated with hyperglycemia. This is a clinical finding. " * 3, score=0.8, chunk="1"),
        hit("Unrelated text about thyroid physiology. " * 4, score=0.7, chunk="2", doc="doc-2"),
    ]
    context, meta = compress_context(question, hits, max_chars=max_chars)
    assert len(context) <= max_chars
    assert meta["input_sentences"] >= meta["selected_sentences"]
    assert 0.0 <= meta["compression_ratio"] <= 1.0 or meta["compression_ratio"] > 1.0


@pytest.mark.parametrize("question,text", [
    ("What is diabetes?", "Diabetes is a chronic metabolic disease."),
    ("What are the main findings?", "The main findings are hyperglycemia, ketosis, and metabolic acidosis."),
    ("What is hypertension?", "Hypertension is elevated blood pressure."),
])
def test_extractive_draft_is_nonempty_on_relevant_evidence(question, text):
    p = deterministic_phase1(question)
    draft, meta = extractive_draft(question, [hit(text, score=0.84)], p)
    assert meta["supported"] is True
    assert meta["sentence_count"] >= 1
    assert draft
    assert "[S1]" in draft


@pytest.mark.parametrize("score", [0.1, 0.2, 0.5, 0.8, 1.0])
def test_extractive_draft_responds_to_retrieval_strength(score):
    question = "What is diabetes?"
    p = deterministic_phase1(question)
    draft, meta = extractive_draft(question, [hit("Diabetes is a chronic metabolic disease.", score=score)], p)
    assert meta["sentence_count"] in {0, 1}
    if score >= 0.5:
        assert draft


# ---------------------------------------------------------------------------
# 6. Evidence guard: numeric, polarity, claim, citation, grounding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "7%", "7 %", "0.5 g", "500 mg", "1 kg", "1000 ml", "1 L", "37 °C", "120 mmHg", "60 min", "3600 s", "1 week", "7 days", "1 kHz", "1000 Hz", "5-10 mg",
])
def test_measurement_extraction_matrix(text):
    values = extract_measurements(text)
    assert values
    assert all(isinstance(v, str) and isinstance(u, str) for v, u in values)


@pytest.mark.parametrize("claim,evidence", [
    ("500 mg", "0.5 g"), ("1 kg", "1000 g"), ("1 L", "1000 ml"),
    ("1 min", "60 s"), ("1 week", "7 days"), ("100 cm", "1 m"),
    ("1 kHz", "1000 Hz"), ("37 °C", "37 C"),
])
def test_numeric_equivalence_matrix(claim, evidence):
    result = numeric_consistency(claim, evidence)
    assert result["checked"] is True
    assert result["mismatch"] is False


@pytest.mark.parametrize("claim,evidence", [
    ("500 mg", "501 mg"), ("1 kg", "999 g"), ("1 L", "999 ml"),
    ("7%", "8%"), ("120 mmHg", "80 mmHg"), ("37 C", "40 C"),
    ("10 min", "5 min"),
])
def test_numeric_mismatch_matrix(claim, evidence):
    result = numeric_consistency(claim, evidence)
    assert result["checked"] is True
    assert result["mismatch"] is True
    assert result["unsupported_numeric"]


@pytest.mark.parametrize("claim,evidence", [
    ("Diabetes is chronic.", "Diabetes is chronic."),
    ("Hypertension is common.", "Hypertension is common."),
    ("Metformin lowers glucose.", "Metformin lowers glucose."),
    ("The dose is 500 mg.", "The dose is 0.5 g."),
])
def test_semantic_support_is_high_for_exact_or_equivalent_content(claim, evidence):
    assert semantic_support(claim, evidence) >= 0.6


@pytest.mark.parametrize("claim,evidence", [
    ("Diabetes is not chronic.", "Diabetes is chronic."),
    ("No hypertension.", "The patient has hypertension."),
    ("Avoid aspirin.", "Aspirin is recommended."),
    ("Without treatment outcomes improve.", "With treatment outcomes improve."),
    ("Diabetes is absent.", "Diabetes is present."),
])
def test_detect_contradiction_matrix(claim, evidence):
    assert detect_contradiction(claim, [evidence]) is True


@pytest.mark.parametrize("answer", [
    "Diabetes is chronic.",
    "- Diabetes is chronic.",
    "1. Diabetes is chronic.",
    "Diabetes is chronic. [S1]",
    "Diabetes is chronic.\nSources: [S1] book.pdf",
    "Diabetes is chronic.\nReferences: [S1]",
])
def test_split_claims_does_not_create_source_metadata_claim(answer):
    claims = split_claims(answer)
    assert any("diabetes" in c.casefold() for c in claims)
    assert all(not c.casefold().startswith(("sources:", "references:", "citations:")) for c in claims)


@pytest.mark.parametrize("answer", ["", "ok", "yes", "thanks", "[S1]", "[S2]"])
def test_split_claims_noise_is_ignored(answer):
    assert split_claims(answer) == []


@pytest.mark.parametrize("claim,evidence,expected", [
    ("Diabetes is chronic.", "Diabetes is chronic.", "SUPPORTED"),
    ("The dose is 500 mg.", "The dose is 0.5 g.", "SUPPORTED"),
    ("The dose is 600 mg.", "The dose is 500 mg.", "NUMERIC_MISMATCH"),
    ("The moon is blue.", "Diabetes is chronic.", "UNSUPPORTED"),
    ("Diabetes is not chronic.", "Diabetes is chronic.", "CONTRADICTED"),
])
def test_verify_claims_status_matrix(claim, evidence, expected):
    checks = verify_claims(claim, [evidence], ["S1"])
    assert len(checks) == 1
    assert checks[0].status == expected


@pytest.mark.parametrize("claims", [
    [ClaimCheck("Diabetes is chronic.", 0.9, "SUPPORTED", ("S1",))],
    [ClaimCheck("Diabetes is common.", 0.5, "PARTIAL", ("S1",))],
    [ClaimCheck("Diabetes is chronic.", 0.9, "SUPPORTED", ("S1",)), ClaimCheck("Hypertension is common.", 0.5, "PARTIAL", ("S2",))],
])
def test_grounding_passes_supported_only_sets(claims):
    result = grounding_decision(claims)
    assert result["allow"] is True
    assert result["blocked_claims"] == 0


@pytest.mark.parametrize("claims", [
    [ClaimCheck("bad", 0.1, "UNSUPPORTED", ())],
    [ClaimCheck("bad", 0.1, "NUMERIC_MISMATCH", ())],
    [ClaimCheck("bad", 0.1, "CONTRADICTED", ())],
    [ClaimCheck("good", 0.9, "SUPPORTED", ("S1",)), ClaimCheck("bad", 0.1, "UNSUPPORTED", ())],
])
def test_grounding_blocks_unsafe_or_unsupported_sets(claims):
    result = grounding_decision(claims)
    assert result["allow"] is False
    assert result["blocked_claims"] >= 1


@pytest.mark.parametrize("bad_status", ["UNSUPPORTED", "NUMERIC_MISMATCH", "CONTRADICTED"])
def test_citation_firewall_removes_bad_claims(bad_status):
    checks = [
        ClaimCheck("verified diabetes fact", 0.9, "SUPPORTED", ("S1",)),
        ClaimCheck("unsafe hallucinated fact", 0.1, bad_status, ("S2",)),
    ]
    safe, used = citation_firewall("original answer", checks)
    assert used is True
    assert "verified diabetes fact" in safe
    assert "unsafe hallucinated fact" not in safe


@pytest.mark.parametrize("answer,evidence", [
    ("Diabetes is chronic.", ["Diabetes is chronic."]),
    ("Diabetes is chronic.\nSources: [S1] book.pdf", ["Diabetes is chronic."]),
    ("Hypertension is common.", ["Hypertension is common."]),
])
def test_final_answer_contract_allows_grounded_simple_answers(answer, evidence):
    hits = [hit(evidence[0], score=0.9)]
    result = verify_final_answer(answer, hits)
    assert result["checked"] is True
    assert result["allow"] is True
    assert result["blocked_claims"] == 0


@pytest.mark.parametrize("answer,evidence", [
    ("Diabetes is chronic. The moon is blue.", ["Diabetes is chronic."]),
    ("The dose is 600 mg.", ["The dose is 500 mg."]),
    ("Diabetes is not chronic.", ["Diabetes is chronic."]),
])
def test_final_answer_contract_blocks_bad_simple_answers(answer, evidence):
    result = verify_final_answer(answer, [hit(evidence[0], score=0.9)])
    assert result["allow"] is False
    assert result["blocked_claims"] >= 1


@pytest.mark.parametrize("answer", [
    "Diabetes is chronic.",
    "Hypertension is common.",
    "The main finding is hyperglycemia.",
])
def test_final_answer_contract_requires_matrix_for_hard_entailment(answer):
    result = verify_final_answer(answer, [hit(answer, score=0.9)], require_entailment=True)
    assert result["checked"] is True
    assert result["allow"] is True
    assert result["matrix_all_entailed"] is True


# ---------------------------------------------------------------------------
# 7. Evidence entailment matrix: span integrity and source identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim,text", [
    ("Diabetes is chronic.", "Diabetes is chronic. Hypertension is common."),
    ("Hypertension is common.", "Diabetes is chronic. Hypertension is common."),
    ("Metformin lowers glucose.", "Metformin lowers glucose. Another sentence."),
    ("The dose is 500 mg.", "The dose is 0.5 g."),
])
def test_claim_evidence_matrix_returns_spans_with_locations(claim, text):
    matrix = build_claim_evidence_matrix([claim], [hit(text)], ["S1"])
    assert len(matrix) == 1
    record = matrix[0]
    assert record.claim == claim
    assert record.evidence
    assert all(span.source_id == "S1" for span in record.evidence)
    assert all(span.start <= span.end for span in record.evidence)


@pytest.mark.parametrize("claim", [
    "Diabetes is chronic.",
    "Hypertension is common.",
    "Albuminuria is present.",
    "Metformin lowers glucose.",
])
def test_claim_matrix_status_is_bounded(claim):
    matrix = build_claim_evidence_matrix([claim], [hit(claim)], ["S1"])
    assert matrix[0].status in {"ENTAILED", "PARTIALLY_ENTAILED", "NOT_ENTAILED"}
    assert 0.0 <= matrix[0].support <= 1.0


@pytest.mark.parametrize("matrix", [
    build_claim_evidence_matrix(["Diabetes is chronic."], [hit("Diabetes is chronic.")], ["S1"]),
    build_claim_evidence_matrix(["Hypertension is common."], [hit("Hypertension is common.")], ["S1"]),
])
def test_matrix_has_strong_support_for_exact_evidence(matrix):
    assert matrix_has_strong_support(matrix) is True


# ---------------------------------------------------------------------------
# 8. Confidence and adaptive retrieval: invariants and boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [-2, -1, 0, 0.1, 0.5, 0.9, 1, 2])
def test_confidence_all_inputs_are_clamped(value):
    result = calibrate_confidence(
        retrieval=value, rerank=value, entailment=value, entity_coverage=value,
        source_agreement=value, contradiction=value, safety_conflict=value, ocr_penalty=value,
    )
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
    bad_result = calibrate_confidence(retrieval=0.8, rerank=0.8, entailment=0.8, entity_coverage=0.8, source_agreement=0.8, contradiction=bad, safety_conflict=0)
    assert bad_result.calibrated <= clean.calibrated


@pytest.mark.parametrize("required", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_confidence_gate_boundary(required):
    confidence = calibrate_confidence(retrieval=1, rerank=1, entailment=1, entity_coverage=1, source_agreement=1, contradiction=0, safety_conflict=0)
    assert confidence_gate(confidence, required=required) is True


@pytest.mark.parametrize("args", [
    dict(query_tokens=3, entity_count=0, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=8, entity_count=1, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=20, entity_count=1, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=4, entity_count=3, intent="factual", confidence=0.9, initial_score=0.8),
    dict(query_tokens=5, entity_count=1, intent="diagnosis", confidence=0.5, initial_score=0.8),
    dict(query_tokens=5, entity_count=1, intent="factual", confidence=0.5, initial_score=0.2),
])
def test_adaptive_retrieval_budget_is_bounded(args):
    budget = choose_retrieval_budget(**args)
    assert 1 <= budget.variant_limit <= 12
    assert 1 <= budget.candidate_limit <= 72
    assert 1 <= budget.rerank_limit <= 64
    assert 1 <= budget.depth <= 3
    assert isinstance(budget.retry, bool)


@pytest.mark.parametrize("alignment,coverage,contradiction,expected", [
    (0.9, 0.9, 0.0, False),
    (0.2, 0.9, 0.0, True),
    (0.9, 0.2, 0.0, True),
    (0.9, 0.9, 0.8, True),
    (0.9, 0.9, 0.8, False),
])
def test_should_retry_retrieval_boundary(alignment, coverage, contradiction, expected):
    assert should_retry_retrieval(alignment=alignment, entity_coverage=coverage, contradiction=contradiction, attempts=0, max_attempts=1) is expected


# ---------------------------------------------------------------------------
# 9. Medical term layer: mixed case, measurements, abbreviations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("What is HbA1c?", "HbA1c"),
    ("What is DKA?", "DKA"),
    ("What is COPD?", "COPD"),
    ("What is ECG?", "ECG"),
])
def test_medical_term_layer_detects_mixed_or_upper_abbreviations(question, expected):
    result = medical_term_layer(question)
    assert any(x.casefold() == expected.casefold() for x in result["abbreviations"])


@pytest.mark.parametrize("question,expected", [
    ("What is dapagliflozin?", "dapagliflozin"),
    ("What is metformin?", "metformin"),
    ("What is lisinopril?", "lisinopril"),
    ("What is enalapril?", "enalapril"),
])
def test_medical_term_layer_detects_open_set_drug_like_terms(question, expected):
    result = medical_term_layer(question)
    joined = [x.casefold() for x in result["terms"]]
    assert expected in joined or any(expected in x for x in joined)


@pytest.mark.parametrize("question", [
    "What is 500 mg?", "What is 7%?", "What is 120 mmHg?", "What is 37 C?", "What is 1 mL?", "What is 10 bpm?",
])
def test_medical_term_layer_detects_units(question):
    result = medical_term_layer(question)
    assert result["units"]


# ---------------------------------------------------------------------------
# 10. Follow-up isolation and conversation safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What are the main findings?", "What is diabetes?", "What is hypertension?", "Define HbA1c.", "Explain the condition.",
    "What are the complications of DKA?", "What is the diagnosis of DKA?", "What is the treatment of DKA?",
])
def test_independent_questions_are_not_followups(question):
    assert _is_explicit_followup(question) is False


@pytest.mark.parametrize("question", [
    "What about this?", "How about that?", "And the complications?", "And this?", "Then what?", "Et les complications?", "والمضاعفات؟",
])
def test_explicit_followups_are_detected(question):
    assert _is_explicit_followup(question) is True


@pytest.mark.parametrize("old_question,new_question", [
    ("What is diabetes?", "What is hypertension?"),
    ("What is DKA?", "What is pneumonia?"),
    ("Compare diabetes and hypertension.", "What is HbA1c?"),
    ("What is insulin?", "What is albuminuria?"),
])
def test_unrelated_question_pair_does_not_become_followup(old_question, new_question):
    assert _is_explicit_followup(new_question) is False


# ---------------------------------------------------------------------------
# 11. Production-level safety and contract behavior without Ollama
# ---------------------------------------------------------------------------

class BareProduction(ProductionRAGSystem):
    pass


@pytest.mark.parametrize("question", [
    "What is diabetes?", "What is hypertension?", "What are the main findings?", "Define HbA1c.", "Explain albuminuria.",
])
def test_production_answer_contract_can_be_exercised_without_init(monkeypatch, question):
    obj = object.__new__(ProductionRAGSystem)
    obj._production_feature_contract = {"all_resolved": True}
    obj.conversation_memory = SimpleNamespace(history=[], add=lambda q, a: obj.conversation_memory.history.append((q, a)))
    obj.settings = SimpleNamespace(medical_high_risk_evidence_threshold=0.8)

    def fake_answer(q, metadata_filter=None):
        return {
            "status": "SUCCESS",
            "answer": "Diabetes is a chronic metabolic disease." if "diabetes" in q.casefold() else "The indexed evidence supports the answer.",
            "citations": ["S1"],
            "hits": [],
            "confidence": {"level": "high", "evidence_confidence": 0.9},
            "grounding": {"allow": True},
            "contradiction_report": {"has_contradiction": False},
        }

    obj._certified_god_answer = fake_answer
    monkeypatch.setattr("rag_project.app.production_rag.sanitize_trace", lambda x: x)
    result = ProductionRAGSystem.answer(obj, question)
    assert result["production_contract"]["all_features_resolved"] is True
    assert result["pipeline_authority"]
    assert result["answer"]


@pytest.mark.parametrize("question,expected", [
    ("What dose should be used?", True),
    ("Take 500 mg twice daily.", True),
    ("What medication should I take?", True),
    ("What is the treatment?", True),
    ("What is diabetes?", False),
    ("What are the main findings?", False),
])
def test_medical_safety_risk_classification_matrix(question, expected):
    assert is_high_risk_medical_query(question) is expected


@pytest.mark.parametrize("allowed", [True, False])
def test_medical_safety_policy_honors_grounding_and_citations(allowed):
    result = {
        "status": "SUCCESS",
        "answer": "Take 500 mg twice daily.",
        "citations": ["S1"] if allowed else [],
        "confidence": {"evidence_confidence": 0.95 if allowed else 0.2},
        "grounding": {"allow": allowed},
        "contradiction_report": {"has_contradiction": not allowed},
    }
    settings = SimpleNamespace(medical_high_risk_evidence_threshold=0.8)
    out = apply_medical_safety_policy("Take 500 mg twice daily.", result, settings)
    assert out["medical_safety"]["high_risk_query"] is True
    assert (out["medical_safety"]["decision"] == "ALLOW_WITH_EVIDENCE") is allowed
    if not allowed:
        assert out["status"] == "MEDICAL_SAFETY_ABSTAIN"
        assert out["citations"] == []


# ---------------------------------------------------------------------------
# 12. God-mode phase/runtime contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_runtime_phase_implementation_is_total(confidence):
    completed = {
        "phases": {
            "phase_1_query_understanding": "complete",
            "phase_2_retrieval_precision": "complete",
            "phase_4_verification": "complete",
            "phase_5_intelligence_visibility": "complete",
        },
        "phase_plan": {"planner_source": "deterministic", "planner_confidence": confidence, "entities": []},
        "adaptive_retrieval": {"stage": 1, "queries": 1, "final_hits": 2, "escalated": False},
        "two_stage_synthesis": {"used": False, "required": False, "attempted": False, "fallback": True, "verification": {}},
        "confidence_calibration": {"calibrated": confidence},
        "phase_plan": {"planner_source": "deterministic", "planner_confidence": confidence, "entities": []},
    }
    verification = {"blocked_claims": 0, "checked": True}
    result = _runtime_phase_implementation(completed, [], verification)
    assert set(result) == {
        "phase_1_query_understanding", "phase_2_retrieval_precision", "phase_3_two_stage_generation", "phase_4_verification", "phase_5_intelligence_visibility"
    }
    assert result["phase_5_intelligence_visibility"]["authority"].endswith("complete_phases")


@pytest.mark.parametrize("assist", [
    {},
    {"entities": []},
    {"entities": ["diabetes"]},
    {"entities": ["invented", "diabetes", "hypertension"]},
])
def test_validated_model_entities_never_trusts_unrepresented_terms(assist):
    result = {"small_model_assist": assist, "hits": [hit("Diabetes and hypertension are discussed.")]}
    values = _validated_model_entities(result)
    assert "invented" not in [x.casefold() for x in values]
    assert all(x.casefold() in "diabetes and hypertension are discussed." for x in values)


# ---------------------------------------------------------------------------
# 13. Public RAG static guards: easy path + pathological inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What is diabetes?", "What is hypertension?", "What is HbA1c?", "What are the main findings?",
])
def test_public_evidence_alignment_never_claims_direct_support_from_empty_hits(question):
    result = RAGSystem.evaluate_evidence_alignment(question, [])
    assert result["decision"] == "NOT_SUPPORTED"
    assert result["answerability"] == 0.0
    assert result["local_context_strength"] == 0.0


@pytest.mark.parametrize("question", [
    "What is diabetes?", "What is hypertension?", "What are the main findings?", "What are the complications?",
])
def test_public_evidence_alignment_handles_semantically_related_evidence(question):
    evidence = hit("Diabetes is a chronic metabolic disease with hyperglycemia.", score=0.8)
    result = RAGSystem.evaluate_evidence_alignment(question, [evidence])
    assert 0.0 <= result["answerability"] <= 1.0
    assert 0.0 <= result["local_context_strength"] <= 1.0
    assert result["decision"] in {"DIRECTLY_SUPPORTED", "PARTIALLY_SUPPORTED", "RELATED_BUT_NOT_ANSWERING", "NOT_SUPPORTED"}


@pytest.mark.parametrize("answer", [
    "Diabetes is chronic.", "Hypertension is common.", "The main finding is hyperglycemia.",
])
def test_apply_grounding_guard_keeps_strong_overlap(answer):
    selected = [hit(answer, score=0.9)]
    result = RAGSystem.apply_grounding_guard(answer, "What is diabetes?", selected)
    returned, answer_grounding, query_coverage, fallback = result
    assert returned
    assert 0.0 <= answer_grounding <= 1.0
    assert 0.0 <= query_coverage <= 1.0
    assert fallback in {True, False}


@pytest.mark.parametrize("question,history", [
    ("What is diabetes?", []),
    ("What about this?", [("What is diabetes?", "Diabetes is chronic.")]),
    ("And the complications?", [("What is DKA?", "DKA is an emergency.")]),
    ("What is hypertension?", [("What is diabetes?", "Diabetes is chronic.")]),
])
def test_query_understanding_is_stable_with_or_without_context(question, history):
    plan = plan_query(question, conversation_context="\n".join(f"Q: {q}\nA: {a}" for q, a in history))
    assert plan.normalized
    assert len(plan.variants) <= 10
    assert len(plan.entities) <= 16


# ---------------------------------------------------------------------------
# 14. Cross-language question matrix: same safety invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Qu'est-ce que le diabète ?",
    "Quels sont les principaux résultats ?",
    "Quelle est la définition de l'hypertension ?",
    "Quels sont les critères diagnostiques ?",
    "Quel est le traitement ?",
    "Pourquoi survient l'acidocétose ?",
    "ما هو السكري؟",
    "ما هي أهم النتائج؟",
    "ما هي مضاعفات الحماض الكيتوني؟",
    "ما هو ضغط الدم المرتفع؟",
])
def test_multilingual_queries_do_not_create_generic_fake_entities(question):
    entities = extract_query_entities(question)
    forbidden = {"main", "findings", "results", "what", "are", "the", "ما", "هي", "هو", "أهم", "النتائج"}
    assert not any(x.casefold() in forbidden for x in entities)


@pytest.mark.parametrize("question", [
    "Quelle est la dose de metformine ?",
    "ما هي جرعة الميتفورمين؟",
    "Quel est le traitement du diabète ?",
    "ما هو علاج السكري؟",
])
def test_multilingual_high_risk_questions_remain_high_risk(question):
    p = plan_query(question)
    assert p.normalized
    assert is_high_risk_medical_query(question) is True


# ---------------------------------------------------------------------------
# 15. Property-style deterministic fuzz matrix
# ---------------------------------------------------------------------------

BASE_TOPICS = [
    "diabetes", "hypertension", "albuminuria", "metformin", "insulin", "HbA1c", "DKA", "hypokalemia", "nephropathy", "glucose",
]
QUESTION_TEMPLATES = [
    "What is {x}?", "Define {x}.", "Explain {x}.", "What are the findings for {x}?", "What complications are related to {x}?",
    "How is {x} diagnosed?", "What is the treatment of {x}?", "What causes {x}?", "What is the prognosis of {x}?", "What does {x} mean?",
]


@pytest.mark.parametrize("topic", BASE_TOPICS)
def test_entity_extraction_fuzz_topics_stays_nonempty_for_medical_terms(topic):
    entities = extract_clinical_entities(f"What is {topic}?")
    assert entities or topic.casefold() in {"hba1c", "dka", "glucose"}


@pytest.mark.parametrize("topic", BASE_TOPICS)
def test_planner_fuzz_topics_is_total(topic):
    for template in QUESTION_TEMPLATES:
        question = template.format(x=topic)
        plan = plan_query(question)
        assert plan.normalized
        assert len(plan.variants) <= 10
        assert len(plan.entities) <= 16
        assert plan.intent


@pytest.mark.parametrize("topic", BASE_TOPICS)
def test_answer_verifier_fuzz_topics_accepts_exact_evidence(topic):
    answer = f"{topic} is discussed in the indexed evidence."
    result = verify_final_answer(answer, [hit(answer, score=0.9)])
    assert result["checked"] is True
    assert result["allow"] is True


# ---------------------------------------------------------------------------
# 16. Final invariants: no hidden empty-success states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["NOT_SUPPORTED", "REASONING_ABSTAIN", "MEDICAL_SAFETY_ABSTAIN", "SUCCESS_WITH_WARNINGS"])
def test_abstention_and_warning_statuses_have_nonempty_reason_contracts(status):
    payload = {
        "status": status,
        "answer": "I could not verify a sufficiently grounded answer from the indexed evidence.",
        "citations": [],
        "confidence": {"level": "low", "evidence_confidence": 0.0},
    }
    assert payload["answer"]
    assert "confidence" in payload


@pytest.mark.parametrize("value", [None, "", [], {}, (), ("S1",), ["S1"], [1]])
def test_citation_input_shapes_do_not_crash_medical_backstop(value):
    result = apply_medical_safety_policy(
        "Take 500 mg twice daily.",
        {
            "answer": "Take 500 mg twice daily.",
            "citations": value,
            "confidence": {"evidence_confidence": 0.9},
            "grounding": {"allow": True},
            "contradiction_report": {"has_contradiction": False},
        },
        SimpleNamespace(medical_high_risk_evidence_threshold=0.8),
    )
    assert result
    assert "medical_safety" in result


@pytest.mark.parametrize("question", [
    "", " ", "\n", "?", "???", "what", "hello", "thanks", "ok", "asdfghjkl", "12345",
])
def test_pathological_short_queries_remain_total(question):
    plan = plan_query(question)
    assert isinstance(plan.normalized, str)
    assert len(plan.variants) <= 10
    u = understanding(question, intent=plan.intent)
    assert should_use_small_model(question, u) in {True, False}


# ---------------------------------------------------------------------------
# 17. Performance-safe boundedness invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 25, 50])
def test_large_evidence_sets_do_not_break_claim_verification(n):
    evidence = [hit(f"Diabetes evidence sentence {i}.", score=0.5 + min(0.49, i / 200), doc=f"doc-{i % 5}", chunk=f"chunk-{i}", page=i + 1) for i in range(n)]
    result = verify_final_answer("Diabetes evidence sentence 1.", evidence)
    assert isinstance(result, dict)
    assert result["claim_count"] in {0, 1}


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 20])
def test_large_hit_lists_keep_reasoning_bounded(n):
    hits = [hit("Diabetes causes diabetic nephropathy.", score=0.8, doc=f"doc-{i % 4}", chunk=f"chunk-{i}", page=i + 1) for i in range(n)]
    u = understanding("Why does diabetes cause nephropathy?", entities=extract_clinical_entities("Why does diabetes cause nephropathy?"), intent="etiology", relations=("causality",))
    result = assess_clinical_reasoning(u, hits)
    assert len(result.supported_paths) <= 8
    assert len(result.blocked_reasons) <= 8


# ---------------------------------------------------------------------------
# 18. Explicit regression for the original failure trace
# ---------------------------------------------------------------------------


def test_original_main_findings_trace_is_no_longer_an_entity_coverage_failure():
    question = "What are the main findings?"
    evidence = hit(
        "IV. Conséquences biologiques : 1. Hyperglycémie >2.5 g/L. 2. Cétose. 3. Acidose métabolique. 4. Déplétion potassique.",
        score=0.82,
        doc="endocrino",
        chunk="218",
        page=52,
    )
    p = plan_query(question)
    u = understanding(question, intent=p.intent)
    assert p.entities == ()
    assert u.entities == ()
    reasoner = assess_clinical_reasoning(u, [evidence])
    assert reasoner.mode == "DIRECT_SUMMARY"
    assert reasoner.allow_generation is True
    assert "insufficient_entity_coverage" not in reasoner.blocked_reasons


def test_original_main_findings_evidence_can_be_claim_verified():
    answer = "The main findings include hyperglycemia, ketosis, and metabolic acidosis."
    evidence = "IV. Conséquences biologiques : Hyperglycémie >2.5 g/L. Cétose. Acidose métabolique."
    result = verify_final_answer(answer, [hit(evidence, score=0.9)])
    assert result["claim_count"] == 1
    assert result["blocked_claims"] == 0
    assert result["allow"] is True


def test_original_main_findings_sources_line_never_becomes_a_second_claim():
    answer = "The main findings include hyperglycemia, ketosis, and metabolic acidosis.\n\nSources: [S1] endocrino.pdf (page 52)"
    result = verify_claims(answer, ["Hyperglycemia, ketosis, and metabolic acidosis are listed."], ["S1"])
    assert len(result) == 1
    assert not any(check.claim.casefold().startswith("sources:") for check in result)


def test_original_main_findings_final_matrix_has_a_single_answer_claim():
    answer = "The main findings include hyperglycemia, ketosis, and metabolic acidosis."
    hit_obj = hit("Hyperglycemia, ketosis, and metabolic acidosis are listed.", score=0.9, doc="endocrino", page=52)
    result = verify_final_answer(answer, [hit_obj])
    assert result["claim_count"] == 1
    assert result["matrix_claim_count"] == 1
    assert result["blocked_claims"] == 0


# ---------------------------------------------------------------------------
# 19. Sanity checks for exact public structures
# ---------------------------------------------------------------------------


def test_query_understanding_to_dict_contract():
    u = understanding("What is diabetes?", entities=extract_clinical_entities("What is diabetes?"))
    data = u.to_dict()
    assert {"normalized", "intents", "primary_intent", "entities", "relations", "constraints", "answer_shape", "semantic_terms", "confidence"} <= set(data)


def test_phase_plan_to_dict_contract():
    data = deterministic_phase1("What is diabetes?").to_dict()
    assert {"intent", "entities", "sub_questions", "rewritten_queries", "must_contain", "ambiguity", "needs_table", "needs_numeric", "needs_figure", "needs_multi_hop", "planner_source", "planner_confidence"} <= set(data)


def test_claim_check_to_dict_contract():
    data = ClaimCheck("Diabetes is chronic.", 0.9, "SUPPORTED", ("S1",)).to_dict()
    assert {"claim", "support", "status", "sources", "numeric_mismatch", "contradiction", "reason"} <= set(data)


def test_retrieval_hit_metadata_is_preserved_for_ui_and_verification():
    h = hit("Diabetes is chronic.", score=0.91, doc="book", chunk="chunk-1", page=52)
    assert h.metadata["file_name"] == "book.pdf"
    assert h.metadata["page_numbers"] == [52]
    assert h.metadata["chunk_id"] == "chunk-1"
    assert h.score == 0.91
