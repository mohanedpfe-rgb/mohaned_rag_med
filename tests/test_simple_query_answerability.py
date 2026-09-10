from types import SimpleNamespace

from rag_project.intelligence.query_intelligence import extract_query_entities, plan_query
from rag_project.intelligence.top_level_pipeline import complete_phases, deterministic_phase1, extractive_draft, medical_term_layer


def _hit(text: str, score: float = 0.70):
    return SimpleNamespace(
        text=text,
        score=score,
        metadata={"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]},
    )


def test_generic_question_does_not_create_fake_entities():
    assert extract_query_entities("What are the main findings?") == ()
    assert plan_query("What are the main findings?").entities == ()


def test_generic_question_is_not_forced_into_hard_query_path():
    phase = deterministic_phase1("What are the main findings?")
    assert phase.intent == "factual"
    assert phase.needs_multi_hop is False
    assert phase.needs_numeric is False
    assert phase.needs_table is False
    assert phase.needs_figure is False


def test_medical_term_layer_does_not_label_ordinary_words_as_medical_terms():
    result = medical_term_layer("What are the main findings?")
    assert "main" not in result["terms"]
    assert "findings" not in result["terms"]


def test_strong_retrieval_support_can_seed_an_extractive_answer_without_word_overlap():
    phase = deterministic_phase1("What are the main findings?")
    hit = _hit("Conséquences biologiques : hyperglycémie, cétose, acidose métabolique et déplétion potassique.")
    draft, state = extractive_draft("What are the main findings?", [hit], phase)
    assert state["supported"] is True
    assert "hyperglycémie" in draft
    assert "[S1]" in draft


def test_complete_phases_returns_answer_for_simple_grounded_question_without_llm():
    hit = _hit("Conséquences biologiques : hyperglycémie, cétose, acidose métabolique et déplétion potassique.")
    result = complete_phases(None, "What are the main findings?", {"hits": [hit], "answer": ""})
    assert result.get("generation_path") == "extractive_verified_fallback"
    assert result.get("answer")
    assert "hyperglycémie" in result["answer"]
    assert result.get("two_stage_policy", {}).get("required") is False
