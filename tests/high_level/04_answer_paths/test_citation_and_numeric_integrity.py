from __future__ import annotations

import re

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_successful_answer__has_valid_citations_and_grounding_on_exact_extractive_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert str(result.get("answer") or "").strip()
    assert result.get("hits")
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_successful_answer__every_sentence_has_a_valid_source_marker_on_exact_extractive_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    answer = str(result.get("answer") or "").strip()
    sentences = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", answer) if len(part.strip()) >= 18]
    assert sentences
    assert all(re.search(r"\[S\d+\]\s*$", sentence, flags=re.I) for sentence in sentences), answer
    assert_citations_valid(result)


@pytest.mark.high_level
def test_numeric_answer__reports_no_numeric_verification_mismatch_on_exact_template_path(clean_system):
    result = clean_system.answer("What numeric information is stated about HbA1c?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    verification = result.get("verification") or {}
    assert verification.get("numeric_mismatch") is not True, verification
    assert_grounded(result)
