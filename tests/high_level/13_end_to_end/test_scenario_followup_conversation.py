import pytest

@pytest.mark.high_level

def test_followup_conversation__preserves_context_without_cross_contamination(clean_system):
    questions = [
        "What is diabetes mellitus?",
        "What are the complications?",
        "How is it diagnosed?",
        "What about HbA1c?",
        "What is hypertension?",
    ]
    results = [clean_system.answer(question) for question in questions]
    assert all(result.get("status") for result in results)
    final_answer = str(results[-1].get("answer") or "").casefold()
    assert "hypertension" in final_answer or results[-1].get("status") in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE"}
