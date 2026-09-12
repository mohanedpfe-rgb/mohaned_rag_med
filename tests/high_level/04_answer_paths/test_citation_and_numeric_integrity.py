from __future__ import annotations

import re

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_successful_answer__has_valid_citations_and_grounding(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert str(result.get("answer") or "").strip()
    assert result.get("hits"), "successful answer must retain retrieved evidence"
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_successful_answer__every_sentence_has_a_valid_source_marker(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    answer = str(result.get("answer") or "").strip()
    sentences = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", answer) if len(part.strip()) >= 18]
    assert sentences, f"no answer sentences found: {answer!r}"
    assert all(re.search(r"\[S\d+\]\s*$", sentence, flags=re.I) for sentence in sentences), answer
    assert_citations_valid(result)


@pytest.mark.high_level
def test_numeric_answer__reports_no_numeric_verification_mismatch(clean_system):
    result = clean_system.answer("What numeric information is stated about HbA1c?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED"})
    verification = result.get("verification") or {}
    assert verification.get("numeric_mismatch") is not True, verification
    if str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_grounded(result)
