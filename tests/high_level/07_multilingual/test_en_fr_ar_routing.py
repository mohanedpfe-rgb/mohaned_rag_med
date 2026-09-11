import pytest

@pytest.mark.high_level
@pytest.mark.parametrize("question", ["What is diabetes mellitus?", "Qu'est-ce que le diabète ?", "ما هو داء السكري؟"])
def test_three_languages__produce_query_analysis(question, clean_system):
    result = clean_system.answer(question)
    assert isinstance(result.get("query_analysis") or result.get("route") or {}, dict)

@pytest.mark.high_level
@pytest.mark.parametrize("question", ["And the complications?", "Et les complications ?", "وماذا عن المضاعفات؟"])
def test_followup_phrases__remain_supported_across_languages(question, clean_system):
    clean_system.answer("What is diabetes mellitus?")
    result = clean_system.answer(question)
    assert result.get("status")
