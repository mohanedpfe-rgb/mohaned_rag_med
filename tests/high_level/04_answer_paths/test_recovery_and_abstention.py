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
def test_weak_generation__unsupported_llm_claims_are_blocked_and_verified_fallback_survives(clean_system, fake_ollama_fast):
    """A bad synthesis must never leak hallucinated medical claims into the final answer."""
    fake_ollama_fast.response = (
        "Diabetes mellitus is caused by a fictional X-factor and is always cured by one specific drug. [S1]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management implications of diabetes mellitus using the indexed evidence."
    )

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "GENERATION_ABSTAIN"})
    assert fake_ollama_fast.calls, "a hard supported query must attempt constrained synthesis"
    answer = str(result.get("answer") or "").casefold()
    assert "fictional x-factor" not in answer
    assert "always cured by one specific drug" not in answer

    status = str(result.get("status") or "").upper()
    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert result.get("generation_path") in {"PATH_A_VERIFIED_FALLBACK", "PATH_HYBRID_FALLBACK"}, result
        assert result.get("citations"), result
        assert_grounded(result)
        assert_citations_valid(result)
    else:
        assert_abstained(result)


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
