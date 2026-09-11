import pytest

@pytest.mark.high_level

def test_insufficient_evidence__cleanly_abstains(clean_system):
    result = clean_system.answer("What is the cure for xylomediasis?")
    assert str(result.get("status") or "").upper() in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE", "SYSTEM_NOT_READY"}
    assert "xylomediasis" not in str(result.get("answer") or "").casefold() or "no" in str(result.get("answer") or "").casefold()
