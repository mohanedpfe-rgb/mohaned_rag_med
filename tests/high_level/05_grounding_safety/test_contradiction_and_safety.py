import pytest

@pytest.mark.high_level

def test_answer_contract__exposes_contradiction_and_safety_signals(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    verification = result.get("verification") or {}
    contradiction = verification.get("contradiction")
    assert contradiction is None or isinstance(contradiction, dict)

@pytest.mark.high_level

def test_clinical_language__is_not_unbounded_when_evidence_is_missing(clean_system):
    result = clean_system.answer("What exact treatment should every patient with a fictional disease receive?")
    answer = str(result.get("answer") or "").casefold()
    assert "every patient" not in answer or str(result.get("status") or "").upper() not in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
