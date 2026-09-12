from __future__ import annotations

import pytest

from tests.high_level.helpers import (
    assert_abstained,
    assert_citations_valid,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_memory_unchanged,
    assert_pipeline_authority,
)


@pytest.mark.high_level
def test_abstention__unsupported_question_uses_exact_not_supported_status_and_no_citations(clean_system):
    before = list(clean_system.conversation_memory.history)
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert_exact_status(result, "NOT_SUPPORTED")
    assert not result.get("generation_path"), result
    assert_abstained(result)
    assert result.get("citations") == []
    assert result.get("needs_review") is not False
    assert_memory_unchanged(before, clean_system.conversation_memory.history)


@pytest.mark.high_level
def test_abstention__unsupported_question_does_not_invent_claims_or_evidence(clean_system):
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert_exact_status(result, "NOT_SUPPORTED")
    assert not result.get("claims")
    assert not result.get("hits")
    assert not result.get("citations")
    assert_abstained(result)


@pytest.mark.high_level
def test_abstention__blocked_llm_synthesis_is_exactly_withheld_after_constrained_generation(clean_system, fake_ollama_fast):
    before = list(clean_system.conversation_memory.history)
    fake_ollama_fast.response = (
        "Diabetes mellitus is caused by a fictional X-factor and is always cured by one specific drug. [S1]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management implications of diabetes mellitus using the indexed evidence."
    )

    assert_exact_status(result, "GENERATION_ABSTAIN")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    assert fake_ollama_fast.calls
    assert_abstained(result)
    verification = result.get("verification") or {}
    assert int(verification.get("blocked_claims", 0)) > 0
    answer = str(result.get("answer") or "").casefold()
    assert "fictional x-factor" not in answer
    assert "always cured by one specific drug" not in answer
    assert_memory_unchanged(before, clean_system.conversation_memory.history)
    assert_pipeline_authority(result) if result.get("pipeline_authority") else None


@pytest.mark.high_level
def test_recovery__verified_generation_fallback_is_reserved_for_infrastructure_failure(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert result.get("answer")
    assert_grounded(result)
    assert_citations_valid(result)
