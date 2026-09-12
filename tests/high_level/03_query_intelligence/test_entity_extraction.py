from __future__ import annotations

import pytest


@pytest.mark.high_level
def test_query_intelligence__extracts_real_medical_entities(clean_system):
    result = clean_system.answer("What are the contraindications of metformin in type 2 diabetes with renal failure?")
    route = result.get("route") or {}
    entities = [str(item).casefold() for item in route.get("entities", [])]
    assert any("metformin" in entity for entity in entities)
    assert any("diabetes" in entity for entity in entities)
    assert any("renal" in entity or "kidney" in entity for entity in entities)


@pytest.mark.high_level
def test_query_intelligence__does_not_promote_function_words_to_entities(clean_system):
    result = clean_system.answer("What is the relationship between HbA1c and diabetes?")
    entities = [str(item).casefold() for item in (result.get("route") or {}).get("entities", [])]
    banned = {"what", "is", "the", "between", "and", "about", "for"}
    assert not any(entity in banned for entity in entities)


@pytest.mark.high_level
def test_query_intelligence__does_not_invent_entities_not_present_in_user_question(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    route = result.get("route") or {}
    entities = [str(item).casefold() for item in route.get("entities", [])]
    assert any("diabetes" in entity for entity in entities)
    assert not any(token in entity for entity in entities for token in ("metformin", "hba1c", "renal", "insulin")), entities
