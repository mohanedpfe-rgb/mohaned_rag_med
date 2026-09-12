from __future__ import annotations

import pytest

from tests.high_level.helpers import (
    assert_abstained,
    assert_citations_valid,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
)


@pytest.mark.high_level
def test_abstention__unsupported_question_uses_exact_not_supported_path_and_no_citations(clean_system):
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert_exact_status(result, "ABSTAIN")
    assert not result.get("generation_path"), result
    assert_abstained(result)
    assert result.get("citations") == []
    assert result.get("needs_review") is not False


@pytest.mark.high_level
def test_abstention__unsupported_question_does_not_invent_claims_or_evidence(clean_system):
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert_exact_status(result, "ABSTAIN")
    assert not result.get("claims")
    assert not result.get("hits")
    assert not result.get("citations")
    assert_abstained(result)


@pytest.mark.high_level
def test_recovery__primary_bad_synthesis_falls_to_exact_verified_extractive_path(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = (
        "Diabetes mellitus is caused by a fictional X-factor and is always cured by one specific drug. [S1]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management implications of diabetes mellitus using the indexed evidence."
    )

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_VERIFIED_FALLBACK")
    assert fake_ollama_fast.calls
    answer = str(result.get("answer") or "").casefold()
    assert "fictional x-factor" not in answer
    assert "always cured by one specific drug" not in answer
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_recovery__supported_simple_answer_remains_exact_extractive_success(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert result.get("answer")
    assert_grounded(result)
    assert_citations_valid(result)
