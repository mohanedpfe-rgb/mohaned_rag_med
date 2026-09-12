from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status


@pytest.mark.high_level
def test_query_intelligence__extracts_metformin_entity_on_exact_extractive_answer_path(clean_system):
    result = clean_system.answer("What is metformin used for?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    entities = [str(item).casefold() for item in (result.get("route") or {}).get("entities", [])]
    assert any("metformin" in entity for entity in entities)
    assert any("diabetes" in entity for entity in entities)


@pytest.mark.high_level
def test_query_intelligence__does_not_promote_function_words_to_entities_on_exact_extractive_path(clean_system):
    result = clean_system.answer("What is the relationship between HbA1c and diabetes?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    entities = [str(item).casefold() for item in (result.get("route") or {}).get("entities", [])]
    banned = {"what", "is", "the", "between", "and", "about", "for"}
    assert not any(entity in banned for entity in entities)


@pytest.mark.high_level
def test_query_intelligence__does_not_invent_unasked_entities_on_exact_extractive_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    entities = [str(item).casefold() for item in (result.get("route") or {}).get("entities", [])]
    assert any("diabetes" in entity for entity in entities)
    assert not any(token in entity for entity in entities for token in ("metformin", "hba1c", "renal", "insulin")), entities
