from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_grounded


@pytest.mark.high_level
def test_multilingual__english_french_arabic_questions_route_to_real_answer_paths(clean_system):
    questions = {
        "en": "What is diabetes mellitus?",
        "fr": "Qu'est-ce que le diabète mellitus ?",
        "ar": "ما هو داء السكري؟",
    }
    for language, question in questions.items():
        result = clean_system.answer(question)
        status = str(result.get("status") or "").upper()
        assert status in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}, language
        route = result.get("route") or {}
        analysis = result.get("query_analysis") or {}
        assert route or analysis
        assert result.get("pipeline_authority") or result.get("canonical_pipeline_executed") is True
        assert route.get("language") == language, (language, route)
        assert 0.0 <= float(route.get("language_confidence", -1.0)) <= 1.0
        assert (result.get("query_trace") or {}).get("language") == language
        if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
            assert_grounded(result)
            assert_citations_valid(result)


@pytest.mark.high_level
def test_multilingual__language_queries_retrieve_indexed_medical_evidence(clean_system, ready_multilingual_docs):
    for document in ready_multilingual_docs.values():
        ingestion = clean_system.ingest_file(document)
        assert str(ingestion.get("status") or "").upper() == "READY"

    queries = [
        ("en", "What is chronic diabetes?"),
        ("fr", "Quelle est la définition du diabète ?"),
        ("ar", "ما هو مرض السكري؟"),
    ]
    for language, query in queries:
        result = clean_system.answer(query)
        hits = result.get("hits") or []
        assert hits, f"cross-language query returned no evidence: {query}"
        route = result.get("route") or {}
        assert route.get("language") == language
        text = " ".join(str(getattr(hit, "text", "")) for hit in hits)
        assert any(token in text.casefold() for token in ("diabetes", "diabète", "السكري"))
        status = str(result.get("status") or "").upper()
        if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
            assert_grounded(result)
            assert_citations_valid(result)


@pytest.mark.high_level
def test_multilingual__english_query_retrieves_french_and_arabic_evidence_across_languages(clean_system, ready_multilingual_docs, tmp_path):
    french = write_minimal_pdf(tmp_path / "fr_cross_language.pdf", [
        "Le diabète est une maladie métabolique chronique avec hyperglycémie persistante."
    ])
    arabic = write_minimal_pdf(tmp_path / "ar_cross_language.pdf", [
        "داء السكري هو اضطراب استقلابي مزمن يتميز بارتفاع سكر الدم."
    ])
    assert str(clean_system.ingest_file(french).get("status") or "").upper() == "READY"
    assert str(clean_system.ingest_file(arabic).get("status") or "").upper() == "READY"

    result = clean_system.answer("What does the indexed literature say about chronic diabetes?")
    hits = result.get("hits") or []
    assert hits
    text = " ".join(str(getattr(hit, "text", "")) for hit in hits).casefold()
    assert "diabète" in text or "السكري" in text, "English query did not retrieve non-English medical evidence"
    route = result.get("route") or {}
    assert route.get("language") == "en"
    status = str(result.get("status") or "").upper()
    assert status in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}
    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_grounded(result)
        assert_citations_valid(result)
