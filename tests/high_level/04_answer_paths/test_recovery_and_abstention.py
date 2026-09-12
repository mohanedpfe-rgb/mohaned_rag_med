from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_abstained, assert_status


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
def test_safe_recovery__never_returns_empty_answer_on_supported_question(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    answer = str(result.get("answer") or "").strip()
    assert answer, result
    assert result.get("generation_path"), result
    assert result.get("verification") is not None
