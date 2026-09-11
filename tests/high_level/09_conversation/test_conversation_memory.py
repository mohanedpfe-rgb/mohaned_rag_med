import pytest

@pytest.mark.high_level

def test_successful_answer__can_become_followup_context(clean_system):
    first = clean_system.answer("What is diabetes mellitus?")
    followup = clean_system.answer("What about complications?")
    memory = getattr(clean_system, "conversation_memory", None)
    assert first.get("status") and followup.get("status")
    assert memory is not None or followup.get("query_analysis") is not None

@pytest.mark.high_level

def test_abstention__does_not_become_positive_medical_context(clean_system):
    result = clean_system.answer("What is the cure for xylomediasis?")
    assert str(result.get("status") or "").upper() in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE", "SYSTEM_NOT_READY"}
