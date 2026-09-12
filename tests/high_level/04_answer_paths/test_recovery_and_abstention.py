from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_abstained, assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_unsupported_question__returns_safe_abstention_without_citations(clean_system):
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert_status(result, {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE"})
    assert_abstained(result)
    assert result.get("citations") == []
    assert result.get("needs_review") is not False


@pytest.mark.high_level
def test_unsupported_question__does_not_invent_evidence(clean_system):
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert_status(result, {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE"})
    assert not (result.get("claims") or [])
    answer = str(result.get("answer") or "").casefold()
    assert "xylomediasis" not in answer or "could not" in answer or "insufficient" in answer


@pytest.mark.high_level
def test_weak_generation__unsupported_llm_claims_are_blocked_and_answer_abstains(clean_system, fake_ollama_fast):
    """Retrieved evidence must not be enough to let an unconstrained model invent facts."""
    fake_ollama_fast.response = (
        "Diabetes mellitus is caused by a fictional X-factor and is always cured by one specific drug. [S1]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management implications of diabetes mellitus using the indexed evidence."
    )

    assert_status(result, {"GENERATION_ABSTAIN", "NOT_SUPPORTED", "ANSWER_UNAVAILABLE"})
    assert fake_ollama_fast.calls, "a hard supported query must attempt constrained synthesis before failing closed"
    assert not result.get("citations"), result
    verification = result.get("verification") or {}
    assert verification.get("allow") is False or str(result.get("status") or "").upper() != "SUCCESS"
    assert_abstained(result)
    assert "fictional x-factor" not in str(result.get("answer") or "").casefold()


@pytest.mark.high_level
def test_safe_recovery__never_returns_empty_answer_on_supported_question(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    answer = str(result.get("answer") or "").strip()
    assert answer, result
    assert result.get("generation_path"), result
    assert result.get("verification") is not None
    assert_grounded(result)
    assert_citations_valid(result)
