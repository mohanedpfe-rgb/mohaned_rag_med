from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid


@pytest.mark.high_level
def test_e2e_multilingual__english_french_arabic_library_supports_queries(clean_system, ready_multilingual_docs):
    for document in ready_multilingual_docs.values():
        ingestion = clean_system.ingest_file(document)
        assert str(ingestion.get("status") or "").upper() == "READY"

    scenarios = [
        "What is diabetes mellitus?",
        "Qu'est-ce que le diabète ?",
        "ما هو داء السكري؟",
    ]
    for question in scenarios:
        result = clean_system.answer(question)
        assert result.get("hits"), question
        assert str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
        assert_citations_valid(result)
