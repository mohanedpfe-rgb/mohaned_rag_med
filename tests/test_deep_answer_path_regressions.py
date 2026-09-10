from types import SimpleNamespace

from rag_project.intelligence.advanced_clinical_reasoner import assess_clinical_reasoning
from rag_project.intelligence.evidence_guard import grounding_decision, split_claims, verify_claims
from rag_project.intelligence.small_model_reasoner import should_use_small_model
from rag_project.intelligence.semantic_reasoning import QueryUnderstanding
from rag_project.intelligence.top_level_pipeline import deterministic_phase1


def _understanding(question: str, entities=(), intent="factual"):
    return QueryUnderstanding(
        normalized=question.casefold(),
        intents=(intent,),
        primary_intent=intent,
        entities=tuple(entities),
        relations=(),
        constraints=(),
        answer_shape="explanation",
        semantic_terms=tuple(question.casefold().split()),
        confidence=0.5,
    )


def _hit(text: str, score: float = 0.72):
    return SimpleNamespace(
        text=text,
        score=score,
        doc_id="doc-1",
        metadata={"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]},
    )


def test_generic_fact_question_is_not_small_model_escalated():
    u = _understanding("What are the main findings?")
    assert should_use_small_model("What are the main findings?", u) is False


def test_generic_fact_question_is_not_hard_in_deterministic_planner():
    phase = deterministic_phase1("What are the main findings?")
    assert phase.intent == "factual"
    assert phase.needs_multi_hop is False
    assert phase.needs_numeric is False
    assert phase.needs_table is False
    assert phase.needs_figure is False


def test_entity_free_factual_reasoning_can_be_direct_summary():
    u = _understanding("What are the main findings?")
    result = assess_clinical_reasoning(
        u,
        [_hit("Les principales conséquences biologiques sont l'hyperglycémie, la cétose et l'acidose métabolique.")],
    )
    assert result.mode == "DIRECT_SUMMARY"
    assert result.allow_generation is True
    assert "insufficient_entity_coverage" not in result.blocked_reasons


def test_source_metadata_is_not_treated_as_a_claim():
    claims = split_claims("Hyperglycémie observée.\n\nSources: [S1] endocrine.pdf (pages [52])")
    assert claims == ["Hyperglycémie observée."]


def test_source_line_cannot_block_grounding_when_claim_is_supported():
    checks = verify_claims(
        "Hyperglycémie observée.\n\nSources: [S1] endocrine.pdf",
        ["Hyperglycémie observée."],
        ["S1"],
    )
    decision = grounding_decision(checks)
    assert len(checks) == 1
    assert checks[0].status in {"SUPPORTED", "PARTIAL"}
    assert decision["allow"] is True


def test_entity_free_summary_uses_retrieval_score_as_evidence_signal():
    u = _understanding("What are the main findings?")
    result = assess_clinical_reasoning(u, [_hit("A relevant summary passage.", score=0.8)])
    assert result.direct_support >= 0.72
    assert result.entity_coverage == 1.0


def test_small_model_must_not_invent_hard_intent_for_generic_query():
    u = _understanding("What are the main findings?")
    from rag_project.intelligence.small_model_reasoner import _sanitize_assist

    sanitized = _sanitize_assist(
        "What are the main findings?",
        u,
        {
            "intent": "diagnosis",
            "entities": ["main", "findings"],
            "relations": ["causality"],
            "constraints": ["safety"],
            "subquestions": ["diagnostic criteria"],
            "retrieval_terms": ["diagnosis criteria"],
            "answer_strategy": "diagnostic reasoning",
        },
    )
    assert sanitized["intent"] == "factual"
    assert sanitized["entities"] == []
    assert sanitized["relations"] == []
    assert sanitized["constraints"] == []


def test_advanced_reasoner_still_blocks_missing_entity_for_relationship_query():
    u = _understanding("Why is diabetes related to hypertension?", intent="etiology")
    result = assess_clinical_reasoning(u, [_hit("Unrelated endocrine facts.")])
    assert result.allow_generation is False
    assert "insufficient_entity_coverage" in result.blocked_reasons or "no_explicit_reasoning_path" in result.blocked_reasons
