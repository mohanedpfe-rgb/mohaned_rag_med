from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_multilingual__cross_language_queries_return_exact_grounded_answers_with_language_preserved(clean_system, ready_multilingual_docs):
    for document in ready_multilingual_docs.values():
        result = clean_system.ingest_file(document)
        assert str(result.get("status") or "").upper() == "READY"

    pairs = [
        ("What is diabetes?", "diabetes", "en"),
        ("Quelle est la maladie métabolique chronique ?", "diabète", "fr"),
        ("ما هو الاضطراب الاستقلابي المزمن؟", "السكري", "ar"),
    ]
    for question, expected, language in pairs:
        result = clean_system.answer(question)
        assert_exact_status(result, "SUCCESS")
        assert_exact_path(result, "PATH_A_EXTRACTIVE")
        assert result.get("hits")
        assert_citations_valid(result)
        assert_grounded(result)
        route = result.get("route") or {}
        assert route.get("language") == language
        combined = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
        assert expected.casefold() in combined.casefold()
