import pytest

@pytest.mark.high_level

def test_entity_extraction__keeps_medical_terms(clean_system):
    result = clean_system.answer("What complications are associated with diabetic ketoacidosis?")
    text = str(result.get("query_analysis") or result.get("route") or result)
    assert "diabetic" in text.casefold() or "ketoacidosis" in text.casefold()

@pytest.mark.high_level

def test_entity_extraction__does_not_promote_planner_words(clean_system):
    result = clean_system.answer("What is diabetes?")
    entities = (result.get("query_analysis") or result.get("route") or {}).get("entities", [])
    banned = {"relevant entities", "follow-up", "follow up", "planner"}
    assert not any(str(entity).casefold().strip() in banned for entity in entities)
