from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_e2e_multilingual__english_french_arabic_library_supports_exact_language_routed_answers(clean_system, ready_multilingual_docs):
    for document in ready_multilingual_docs.values():
        ingestion = clean_system.ingest_file(document)
        assert str(ingestion.get("status") or "").upper() == "READY"

    scenarios = [
        ("en", "What is diabetes mellitus?", "diabetes"),
        ("fr", "Qu'est-ce que le diabète ?", "diabète"),
        ("ar", "ما هو داء السكري؟", "السكري"),
    ]
    for language, question, expected in scenarios:
        result = clean_system.answer(question)
        assert_exact_status(result, "SUCCESS")
        assert_exact_path(result, "PATH_A_EXTRACTIVE")
        assert result.get("hits")
        assert_citations_valid(result)
        assert_grounded(result)

        route = result.get("route") or {}
        assert route.get("language") == language
        assert 0.0 <= float(route.get("language_confidence", -1.0)) <= 1.0
        trace = result.get("query_trace") or {}
        assert trace.get("language") == language
        assert trace.get("pipeline_authority") == result.get("pipeline_authority")

        combined = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
        assert expected.casefold() in combined.casefold(), f"{language} query lost its expected evidence"


@pytest.mark.high_level
def test_e2e_multilingual__english_query_retrieves_non_english_diabetes_evidence(clean_system, ready_multilingual_docs):
    for language in ("fr", "ar"):
        ingestion = clean_system.ingest_file(ready_multilingual_docs[language])
        assert str(ingestion.get("status") or "").upper() == "READY"

    result = clean_system.answer("What does the indexed literature say about chronic diabetes?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    text = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
    assert "diabète" in text.casefold() or "السكري" in text.casefold()
    assert_citations_valid(result)
    assert_grounded(result)
