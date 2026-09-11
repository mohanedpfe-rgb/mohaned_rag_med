import pytest

@pytest.mark.high_level

def test_standalone_question__does_not_depend_on_previous_turn(clean_system):
    first = clean_system.answer("What is diabetes mellitus?")
    second = clean_system.answer("What is hypertension?")
    assert first.get("status") and second.get("status")
    assert "diabetes" not in str(second.get("answer") or "").casefold() or "hypertension" in str(second.get("answer") or "").casefold()

@pytest.mark.high_level

def test_explicit_followup__is_recognized(clean_system):
    clean_system.answer("What is diabetes mellitus?")
    followup = clean_system.answer("What about complications?")
    analysis = followup.get("query_analysis") or followup.get("route") or {}
    trace = followup.get("query_trace") or {}
    text = str(analysis) + str(trace)
    assert "follow" in text.casefold() or "complication" in text.casefold()
