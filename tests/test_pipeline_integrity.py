from types import SimpleNamespace

from rag_project.intelligence import entity_coverage, final_answer_contract, god_mode_100, top_level_pipeline
from rag_project.intelligence.pipeline_integrity import (
    install,
    is_control_message,
    safe_extract_query_entities,
    safe_rewrite_follow_up,
    safe_score_entity_coverage,
    safe_verify_final_answer,
)
from rag_project.intelligence.query_intelligence import plan_query


def _hit(text: str, score: float = 0.82):
    return SimpleNamespace(
        text=text,
        score=score,
        doc_id="doc-1",
        metadata={"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]},
    )


def test_standalone_query_is_not_contaminated_by_internal_labels():
    rewritten = safe_rewrite_follow_up("What are the main findings?", [])
    assert rewritten == "What are the main findings?"
    assert "Follow-up:" not in rewritten
    assert "Relevant entities:" not in rewritten


def test_real_followup_is_rewritten_without_protocol_metadata():
    rewritten = safe_rewrite_follow_up(
        "What about this?",
        [("What are the complications of diabetes?", "The document discusses diabetic nephropathy.")],
    )
    assert rewritten
    assert "Follow-up:" not in rewritten
    assert "Relevant entities:" not in rewritten
    assert "diabetes mellitus" in rewritten or "diabetic nephropathy" in rewritten


def test_generic_factual_question_has_no_fake_entities():
    entities = safe_extract_query_entities("What are the main findings?")
    assert entities == ()
    assert "relevant entities" not in entities
    assert "main findings" not in entities


def test_real_medical_entities_are_preserved():
    entities = safe_extract_query_entities(
        "What is the relationship between dapagliflozin and albuminuria?"
    )
    assert any("dapagliflozin" == entity for entity in entities)
    assert any("albuminuria" == entity for entity in entities)


def test_planner_entity_cannot_inject_internal_label():
    entities = safe_extract_query_entities(
        "What are the main findings?",
        planned_entities=("relevant entities", "main findings", "diabetes mellitus"),
    )
    assert "relevant entities" not in entities
    assert "main findings" not in entities
    assert "diabetes mellitus" not in entities


def test_entity_coverage_does_not_penalize_entity_free_summary():
    result = safe_score_entity_coverage(
        "What are the main findings?",
        [_hit("Les principales conséquences biologiques sont l'hyperglycémie, la cétose et l'acidose métabolique.")],
    )
    assert result["entity_count"] == 0
    assert result["coverage"] == 0.0
    assert result["missing"] == []


def test_entity_coverage_recognizes_real_french_evidence():
    result = safe_score_entity_coverage(
        "What is the relationship between dapagliflozin and albuminuria?",
        [_hit("L'albuminurie a été réduite chez les participants traités par dapagliflozine.")],
    )
    assert result["entity_count"] >= 2
    assert not any(entity == "relevant entities" for entity in result["query_entities"])
    # The French drug alias is normalized by the semantic entity extractor.
    assert any("dapagliflozin" == entity for entity in result["query_entities"])
    assert any("albuminuria" == entity for entity in result["query_entities"])


def test_clean_question_does_not_become_multihop():
    query = safe_rewrite_follow_up("What are the main findings?", [])
    plan = plan_query(query)
    assert plan.intent == "factual"
    assert plan.needs_multi_hop is False
    assert plan.needs_numeric is False
    assert plan.needs_table is False
    assert plan.needs_figure is False


def test_abstention_is_control_state_not_medical_claim():
    message = "I could not verify a sufficiently grounded answer from the indexed evidence."
    assert is_control_message(message) is True
    result = safe_verify_final_answer(message, [_hit("Unrelated clinical evidence.")])
    assert result["checked"] is False
    assert result["allow"] is False
    assert result["reason"] == "abstention_not_claim"
    assert result["claim_count"] == 0


def test_real_answer_is_still_verified_as_claims():
    result = safe_verify_final_answer(
        "Hyperglycemia was observed. [S1]",
        [_hit("Hyperglycemia was observed.")],
    )
    assert result["checked"] is True
    assert result["claim_count"] == 1


def test_install_patches_all_canonical_references():
    install()
    assert top_level_pipeline.rewrite_follow_up is safe_rewrite_follow_up
    assert entity_coverage.extract_query_entities is safe_extract_query_entities
    assert entity_coverage.score_entity_coverage is safe_score_entity_coverage
    assert final_answer_contract.verify_final_answer is safe_verify_final_answer
    assert god_mode_100.score_entity_coverage is safe_score_entity_coverage
    assert god_mode_100.verify_final_answer is safe_verify_final_answer
