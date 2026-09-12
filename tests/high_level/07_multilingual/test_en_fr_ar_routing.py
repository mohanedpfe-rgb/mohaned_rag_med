from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_entities_present, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_multilingual__english_french_arabic_simple_questions_have_exact_success_path(clean_system):
    questions = {
        "en": "What is diabetes mellitus?",
        "fr": "Qu'est-ce que le diabète ?",
        "ar": "ما هو داء السكري؟",
    }
    for language, question in questions.items():
        result = clean_system.answer(question)
        assert_exact_status(result, "SUCCESS")
        assert_exact_path(result, "PATH_A_EXTRACTIVE")
        route = result.get("route") or {}
        assert route.get("language") == language, (language, route)
        assert 0.0 <= float(route.get("language_confidence", -1.0)) <= 1.0
        assert (result.get("query_trace") or {}).get("language") == language
        assert_grounded(result)
        assert_citations_valid(result)


@pytest.mark.high_level
def test_multilingual__english_query_retrieves_french_and_arabic_evidence(clean_system, tmp_path):
    french = write_minimal_pdf(tmp_path / "fr_cross_language.pdf", [
        "Le diabète est une maladie métabolique chronique avec hyperglycémie persistante."
    ])
    arabic = write_minimal_pdf(tmp_path / "ar_cross_language.pdf", [
        "داء السكري هو اضطراب استقلابي مزمن يتميز بارتفاع سكر الدم."
    ])
    assert str(clean_system.ingest_file(french).get("status") or "").upper() == "READY"
    assert str(clean_system.ingest_file(arabic).get("status") or "").upper() == "READY"

    result = clean_system.answer("What does the indexed literature say about chronic diabetes?")
    assert_exact_status(result, "SUCCESS")
    route = result.get("route") or {}
    assert route.get("language") == "en"
    hits = result.get("hits") or []
    assert hits
    text = " ".join(str(getattr(hit, "text", "")) for hit in hits).casefold()
    assert "diabète" in text or "السكري" in text
    assert_grounded(result)
    assert_citations_valid(result)


@pytest.mark.high_level
def test_multilingual__follow_up_preserves_entity_within_french_session(clean_system):
    first = clean_system.answer("Qu'est-ce que le diabète ?")
    assert_exact_status(first, "SUCCESS")
    assert_exact_path(first, "PATH_A_EXTRACTIVE")
    history_before = list(getattr(clean_system.conversation_memory, "history", []) or [])

    follow_up = clean_system.answer("Et sa définition ?")
    assert_exact_status(follow_up, "SUCCESS")
    assert_exact_path(follow_up, "PATH_A_EXTRACTIVE")
    route = follow_up.get("route") or {}
    assert route.get("language") == "fr"
    assert route.get("is_follow_up") is True
    rewritten = str(follow_up.get("rewritten_question") or "").casefold()
    assert "diab" in rewritten
    assert len(getattr(clean_system.conversation_memory, "history", []) or []) == len(history_before) + 1
    assert_grounded(follow_up)


@pytest.mark.high_level
def test_multilingual__arabic_morphology_preserves_diabetes_entity_and_numeric_intent(clean_system):
    result = clean_system.answer("ما هي جرعة الميتفورمين لمرضى السكري من النوع الثاني؟")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    route = result.get("route") or {}
    assert route.get("language") == "ar"
    assert route.get("numeric_sensitivity") is True
    assert_entities_present(result, {"metformin"})
    entities = " ".join(str(item).casefold() for item in route.get("entities", []))
    assert "سكري" in entities or "diabetes" in entities
    assert "500 mg" in str(result.get("answer") or "")
    assert_grounded(result)
    assert_citations_valid(result)
