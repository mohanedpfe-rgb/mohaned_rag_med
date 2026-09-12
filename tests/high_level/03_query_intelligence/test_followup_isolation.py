from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_followup_query__is_detected_as_followup_without_losing_medical_entity(clean_system):
    first = clean_system.answer("What is metformin used for in type 2 diabetes?")
    assert_status(first, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})

    second = clean_system.answer("What about its dose?")
    assert_status(second, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"})

    route = second.get("route") or {}
    assert route.get("is_follow_up") is True
    entities = " ".join(str(item) for item in route.get("entities", []))
    assert "metformin" in entities.casefold() or "metformin" in str(second.get("rewritten_question") or "").casefold()


@pytest.mark.high_level
def test_followup_query__does_not_duplicate_previous_question_in_memory(clean_system):
    clean_system.answer("What is diabetes mellitus?")
    clean_system.answer("What about HbA1c?")

    memory = getattr(clean_system, "conversation_memory", None)
    assert memory is not None
    prompt_context = str(memory.prompt_context()).casefold()
    assert prompt_context.count("what is diabetes mellitus?") <= 1
    assert "hba1c" in prompt_context
