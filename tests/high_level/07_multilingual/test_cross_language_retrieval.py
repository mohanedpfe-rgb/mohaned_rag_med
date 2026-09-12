from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid


@pytest.mark.high_level
def test_multilingual__cross_language_queries_return_medical_hits(clean_system, ready_multilingual_docs):
    for document in ready_multilingual_docs.values():
        result = clean_system.ingest_file(document)
        assert str(result.get("status") or "").upper() == "READY"

    pairs = [
        ("What is diabetes?", "diabetes"),
        ("Quelle est la maladie métabolique chronique ?", "diabète"),
        ("ما هو الاضطراب الاستقلابي المزمن؟", "السكري"),
    ]
    for question, expected in pairs:
        result = clean_system.answer(question)
        assert result.get("hits"), question
        assert_citations_valid(result)
        combined = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
        assert expected.casefold() in combined.casefold()
