import pytest

@pytest.mark.high_level

def test_cross_language_question__is_not_rejected_by_language_alone(clean_system):
    for question in ("What is diabetes mellitus?", "Qu'est-ce que le diabète ?", "ما هو داء السكري؟"):
        result = clean_system.answer(question)
        assert str(result.get("status") or "").upper() not in {"LANGUAGE_UNSUPPORTED"}
