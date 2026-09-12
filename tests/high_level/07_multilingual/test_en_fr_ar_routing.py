from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid


@pytest.mark.high_level
def test_multilingual__english_french_arabic_questions_route_to_real_answer_paths(clean_system):
    questions = {
        "en": "What is diabetes mellitus?",
        "fr": "Qu'est-ce que le diabète mellitus ?",
        "ar": "ما هو داء السكري؟",
    }
    for language, question in questions.items():
        result = clean_system.answer(question)
        assert str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}, language
        route = result.get("route") or {}
        analysis = result.get("query_analysis") or {}
        assert route or analysis
        assert result.get("pipeline_authority") or result.get("canonical_pipeline_executed") is True
        if str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
            assert_citations_valid(result)


@pytest.mark.high_level
def test_multilingual__language_queries_retrieve_indexed_medical_evidence(clean_system, ready_multilingual_docs):
    for document in ready_multilingual_docs.values():
        ingestion = clean_system.ingest_file(document)
        assert str(ingestion.get("status") or "").upper() == "READY"

    queries = [
        "What is chronic diabetes?",
        "Quelle est la définition du diabète ?",
        "ما هو مرض السكري؟",
    ]
    for query in queries:
        result = clean_system.answer(query)
        hits = result.get("hits") or []
        assert hits, f"cross-language query returned no evidence: {query}"
        text = " ".join(str(getattr(hit, "text", "")) for hit in hits)
        assert any(token in text.casefold() for token in ("diabetes", "diabète", "السكري"))
