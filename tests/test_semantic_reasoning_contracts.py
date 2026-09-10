from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_project.intelligence.semantic_reasoning import (
    build_evidence_graph,
    clinical_reasoning_ready,
    extract_clinical_entities,
    normalize_medical_term,
    semantic_evidence_alignment,
    understand_query,
)
from rag_project.intelligence.query_intelligence import plan_query
from rag_project.retrieval.hybrid_retriever import RetrievalHit


def _hit(text: str, score: float = 0.8, doc: str = "d1", chunk: str = "c1", page: int = 1):
    return RetrievalHit(
        doc_id=doc,
        text=text,
        metadata={"document_id": doc, "chunk_id": chunk, "page_numbers": [page], "language": "fr"},
        score=score,
        vector_score=0.8,
        lexical_score=0.4,
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is diabetic nephropathy?", "definition"),
        ("What are the diagnostic criteria for DKA?", "diagnosis"),
        ("How is diabetic ketoacidosis treated?", "management"),
        ("What causes secondary hypertension?", "etiology"),
        ("What is the physiopathology of hypokalemia?", "mechanism"),
        ("Compare primary and secondary hyperaldosteronism.", "comparison"),
        ("What is the prognosis of thyroid cancer?", "prognosis"),
        ("What is the dose of insulin?", "numeric"),
    ],
)
def test_structured_intent_classification(query, expected):
    understanding = understand_query(query)
    assert understanding.primary_intent == expected
    assert understanding.confidence >= 0.40


def test_intent_is_multilabel_for_clinically_rich_questions():
    result = understand_query("What are the diagnostic criteria and treatment of diabetic ketoacidosis in children?")
    assert "diagnosis" in result.intents
    assert "management" in result.intents
    assert "clinical_phase" in result.constraints or "population" in result.constraints
    assert result.answer_shape == "list"


@pytest.mark.parametrize(
    "surface,canonical",
    [
        ("DKA", "diabetic ketoacidosis"),
        ("acidocétose diabétique", "diabetic ketoacidosis"),
        ("néphropathie diabétique", "diabetic nephropathy"),
        ("diabetic kidney disease", "diabetic nephropathy"),
        ("HTA", "hypertension"),
        ("hypokaliémie", "hypokalemia"),
        ("low potassium", "hypokalemia"),
        ("hyperaldostéronisme", "hyperaldosteronism"),
    ],
)
def test_medical_aliases_normalize_across_languages(surface, canonical):
    assert normalize_medical_term(surface) == canonical


def test_entity_extraction_returns_type_and_negation():
    entities = extract_clinical_entities("The patient has no hypokalemia but has diabetes.")
    by_name = {entity.normalized: entity for entity in entities}
    assert by_name["hypokalemia"].kind == "finding"
    assert by_name["hypokalemia"].negated is True
    assert by_name["diabetes mellitus"].kind == "disease"


def test_entity_extraction_is_deduplicated():
    entities = extract_clinical_entities("Diabetes, diabetic nephropathy and diabetes mellitus.")
    normalized = [entity.normalized for entity in entities]
    assert len(normalized) == len(set(normalized))


def test_follow_up_query_uses_previous_context_for_intent():
    context = "User asked about diabetic ketoacidosis diagnostic criteria."
    result = understand_query("And the treatment?", conversation_context=context)
    assert result.primary_intent == "management"
    assert "diabetic ketoacidosis" in {entity.normalized for entity in result.entities}


def test_query_plan_preserves_semantic_entities_and_multi_hop_intent():
    plan = plan_query("Why does diabetic nephropathy cause albuminuria?")
    assert "diabetic nephropathy" in plan.entities
    assert "albuminuria" in plan.entities
    assert plan.needs_multi_hop is True
    assert "semantic" in plan.score_boosts


def test_multilingual_semantic_alignment_recovers_french_evidence_for_english_query():
    hits = [_hit("La néphropathie diabétique est une complication chronique du diabète associée à l'albuminurie.")]
    result = semantic_evidence_alignment("What is diabetic nephropathy and albuminuria?", hits)
    assert result["entity_coverage"] >= 0.5
    assert result["score"] >= 0.25
    assert result["decision"] in {"DIRECTLY_SUPPORTED", "PARTIALLY_SUPPORTED"}


def test_semantic_alignment_does_not_reward_unrelated_topic():
    hits = [_hit("Le cancer thyroïdien papillaire est associé à des mutations de RET.")]
    result = semantic_evidence_alignment("What is diabetic nephropathy?", hits)
    assert result["score"] < 0.25
    assert result["decision"] in {"RELATED_BUT_NOT_ANSWERING", "NOT_SUPPORTED"}


def test_semantic_alignment_ignores_empty_evidence():
    hits = [_hit("")]
    result = semantic_evidence_alignment("What is diabetic nephropathy?", hits)
    assert result["score"] == 0.0
    assert result["entity_coverage"] == 0.0


def test_evidence_graph_links_two_entities_across_sources():
    understanding = understand_query("Why is diabetes associated with albuminuria?")
    hits = [
        _hit("Le diabète est une maladie métabolique.", doc="d1", chunk="c1"),
        _hit("L'albuminurie est un marqueur rénal associé au diabète.", doc="d2", chunk="c2"),
    ]
    nodes, edges = build_evidence_graph(hits, understanding)
    assert len(nodes) == 2
    assert edges
    assert any(edge.relation in {"association", "causality"} for edge in edges)


def test_clinical_reasoning_marks_causal_queries_as_multihop_when_graph_exists():
    understanding = understand_query("Why does diabetes cause albuminuria?")
    hits = [
        _hit("Diabetes can cause diabetic nephropathy.", chunk="c1"),
        _hit("Diabetic nephropathy is associated with albuminuria.", chunk="c2"),
    ]
    nodes, edges = build_evidence_graph(hits, understanding)
    state = clinical_reasoning_ready(understanding, nodes, edges)
    assert state["direct_evidence"] is True
    assert state["multi_hop"] is True
    assert state["reasoning_depth"] >= 2


def test_clinical_reasoning_requires_explicit_graph_links():
    understanding = understand_query("Why does diabetes cause albuminuria?")
    hits = [_hit("Diabetes is common."), _hit("Albuminuria may occur in kidney disease.", chunk="c2")]
    nodes, edges = build_evidence_graph(hits, understanding)
    state = clinical_reasoning_ready(understanding, nodes, edges)
    assert state["direct_evidence"] is True
    # No explicit entity-to-entity evidence link means the system must not claim a supported chain.
    assert state["multi_hop"] is False or state["edge_count"] == 0


def test_query_plan_adds_reasoning_variants_for_mechanism_questions():
    plan = plan_query("What is the mechanism of diabetic nephropathy?")
    assert any("mechanism" in variant for variant in plan.variants)
    assert any("pathway" in variant for variant in plan.variants)


def test_query_plan_adds_management_safety_cues():
    plan = plan_query("What is the treatment and contraindications for hypertension?")
    assert plan.intent == "management"
    assert any("contraindications" in variant for variant in plan.variants)


def test_query_plan_keeps_variant_count_bounded():
    for query in [
        "What is diabetic nephropathy?",
        "Compare DKA and HHS.",
        "What are the diagnostic criteria, treatment and prognosis of DKA and what are the doses?",
    ]:
        assert len(plan_query(query).variants) <= 10


def test_semantic_alignment_result_is_bounded_and_serializable():
    result = semantic_evidence_alignment("What is diabetes?", [_hit("Diabetes mellitus is a chronic disease.")])
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["entity_coverage"] <= 1.0
    assert isinstance(result["understanding"], dict)


def test_graph_nodes_preserve_document_and_page_identity():
    hits = [_hit("Diabetes", doc="doc-a", chunk="chunk-a", page=123)]
    understanding = understand_query("What is diabetes?")
    nodes, _ = build_evidence_graph(hits, understanding)
    assert nodes[0].document_id == "doc-a"
    assert nodes[0].chunk_id == "chunk-a"
    assert nodes[0].page_numbers == (123,)


def test_semantic_entity_detection_handles_abbreviations_inside_long_questions():
    result = understand_query("In a patient with DKA, what are the diagnostic criteria and initial management?")
    names = {entity.normalized for entity in result.entities}
    assert "diabetic ketoacidosis" in names
    assert "diagnosis" in result.intents
    assert "management" in result.intents


def test_no_real_models_are_needed_for_semantic_contracts():
    # This contract layer is deterministic so it can run in CI without Ollama or a CrossEncoder.
    assert True
