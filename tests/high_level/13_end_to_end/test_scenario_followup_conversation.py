from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_e2e_followup__resolved_context_then_topic_switch_is_isolated(clean_system):
    first = clean_system.answer("What is diabetes mellitus?")
    assert_status(first, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})

    followup = clean_system.answer("What about HbA1c?")
    assert str(followup.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}
    assert (followup.get("route") or {}).get("is_follow_up") is True

    switched = clean_system.answer("What is hypertension?")
    assert str(switched.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}
    text = str(switched.get("answer") or "").casefold()
    assert "diabetes" not in text or "hypertension" in text

    memory = clean_system.conversation_memory
    assert len(memory.history) >= 3
    assert memory.history[-1][0] == "What is hypertension?"
