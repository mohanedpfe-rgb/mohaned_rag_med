from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded
from tests.high_level.conftest import write_minimal_pdf


@pytest.mark.high_level
def test_multilingual__french_question_returns_french_evidence_text_on_exact_extractive_path(clean_system, tmp_path):
    french = write_minimal_pdf(tmp_path / "french_answer.pdf", ["Le diabète est une maladie métabolique chronique avec hyperglycémie persistante."])
    assert str(clean_system.ingest_file(french).get("status") or "").upper() == "READY"

    result = clean_system.answer("Qu'est-ce que le diabète ?", {"language": "fr"})

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    answer = str(result.get("answer") or "").casefold()
    assert "diabète" in answer
    assert "metabolic" not in answer
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_multilingual__arabic_question_returns_arabic_evidence_text_on_exact_extractive_path(clean_system, tmp_path):
    arabic = write_minimal_pdf(tmp_path / "arabic_answer.pdf", ["داء السكري هو اضطراب استقلابي مزمن يتميز بارتفاع سكر الدم."])
    assert str(clean_system.ingest_file(arabic).get("status") or "").upper() == "READY"

    result = clean_system.answer("ما هو داء السكري؟", {"language": "ar"})

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    answer = str(result.get("answer") or "")
    assert "داء السكري" in answer or "السكري" in answer
    assert "Diabetes mellitus" not in answer
    assert_citations_valid(result)
    assert_grounded(result)
