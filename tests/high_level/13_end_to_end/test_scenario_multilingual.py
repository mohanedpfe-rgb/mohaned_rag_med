from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded


@pytest.mark.high_level
def test_e2e_multilingual__english_french_arabic_library_supports_queries(clean_system, ready_multilingual_docs):
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
        status = str(result.get("status") or "").upper()
        assert status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}, language
        assert result.get("hits"), question
        assert_citations_valid(result)
        assert_grounded(result)

        combined = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
        assert expected.casefold() in combined.casefold(), f"{language} query lost its cross-language evidence"
        assert result.get("pipeline_authority")
        assert result.get("canonical_pipeline_executed") is True
