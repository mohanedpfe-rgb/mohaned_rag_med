from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status


@pytest.mark.high_level
def test_followup_query__metformin_context_is_preserved_for_exact_dose_path(clean_system):
    first = clean_system.answer("What is metformin used for in type 2 diabetes?")
    assert_exact_status(first, "SUCCESS")
    assert_exact_path(first, "PATH_A_EXTRACTIVE")

    second = clean_system.answer("What about its dose?")
    assert_exact_status(second, "SUCCESS")
    assert_exact_path(second, "PATH_B_TEMPLATE")

    route = second.get("route") or {}
    assert route.get("is_follow_up") is True
    entities = " ".join(str(item) for item in route.get("entities", []))
    rewritten = str(second.get("rewritten_question") or "").casefold()
    assert "metformin" in entities.casefold() or "metformin" in rewritten
    assert "dose" in rewritten
    assert "500 mg" in str(second.get("answer") or "")


@pytest.mark.high_level
def test_followup_query__does_not_duplicate_previous_question_in_memory(clean_system):
    first = clean_system.answer("What is diabetes mellitus?")
    second = clean_system.answer("What about HbA1c?")
    assert_exact_status(first, "SUCCESS")
    assert_exact_status(second, "SUCCESS")

    memory = getattr(clean_system, "conversation_memory", None)
    assert memory is not None
    prompt_context = str(memory.prompt_context()).casefold()
    assert prompt_context.count("what is diabetes mellitus?") == 1
    assert "hba1c" in prompt_context
