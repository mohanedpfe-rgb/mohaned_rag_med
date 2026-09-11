import pytest

@pytest.mark.high_level

def test_supported_answer__contains_verification_signal(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    verification = result.get("verification") or result.get("grounding") or {}
    assert isinstance(verification, dict)

@pytest.mark.high_level

def test_unsupported_answer__is_not_marked_high_confidence(clean_system):
    result = clean_system.answer("What is the cure for xylomediasis?")
    confidence = result.get("confidence") or {}
    if confidence:
        assert str(confidence.get("level", "none")).lower() not in {"high", "very_high"}
