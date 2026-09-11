import pytest
from tests.high_level.helpers import assert_abstained

@pytest.mark.high_level

def test_unsupported_question__abstains(clean_system):
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")
    assert_abstained(result)

@pytest.mark.high_level

def test_recovery_contract__never_returns_empty_answer(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    status = str(result.get("status") or "").upper()
    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert str(result.get("answer") or "").strip()
