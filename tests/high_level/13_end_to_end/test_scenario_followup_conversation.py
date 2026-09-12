from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_e2e_followup__resolved_context_then_topic_switch_is_isolated(clean_system):
    first = clean_system.answer("What is diabetes mellitus?")
    assert_status(first, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert_grounded(first)

    followup = clean_system.answer("What about HbA1c?")
    assert_status(followup, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"})
    assert (followup.get("route") or {}).get("is_follow_up") is True
    rewritten = str(followup.get("rewritten_question") or "").casefold()
    answer = str(followup.get("answer") or "").casefold()
    assert "diabetes" in rewritten or "diabetes" in answer
    assert "hba1c" in rewritten or "hba1c" in answer
    if str(followup.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_citations_valid(followup)
        assert_grounded(followup)

    switched = clean_system.answer("What is hypertension?")
    assert_status(switched, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert_grounded(switched)
    switched_text = str(switched.get("answer") or "").casefold()
    assert "hypertension" in switched_text
    assert "diabetes" not in switched_text or switched_text.index("hypertension") <= switched_text.index("diabetes")

    memory = clean_system.conversation_memory
    assert len(memory.history) >= 3
    assert memory.history[-1][0] == "What is hypertension?"
