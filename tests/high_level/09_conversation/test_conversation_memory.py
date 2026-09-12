from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_status


@pytest.mark.high_level
def test_conversation__successful_answer_is_available_to_real_followup(clean_system):
    first = clean_system.answer("What is metformin used for in type 2 diabetes?")
    assert_status(first, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert_citations_valid(first)

    before_followup = len(clean_system.conversation_memory.history)
    second = clean_system.answer("What about its dose?")
    status = str(second.get("status") or "").upper()
    assert status in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}
    route = second.get("route") or {}
    assert route.get("is_follow_up") is True
    result_text = (str(second.get("rewritten_question") or "") + " " + str(second.get("answer") or "")).casefold()
    assert result_text
    assert "dose" in result_text

    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_citations_valid(second)
        rewritten = str(second.get("rewritten_question") or "").casefold()
        assert "metformin" in rewritten
        assert second.get("hits")

    memory = clean_system.conversation_memory
    assert len(memory.history) >= before_followup + 1
    assert any("metformin" in question.casefold() for question, _ in memory.history)
    assert any("dose" in question.casefold() for question, _ in memory.history)
    assert sum("what about its dose" in question.casefold() for question, _ in memory.history) <= 1


@pytest.mark.high_level
def test_conversation__failed_answer_is_not_persisted(clean_system):
    before = len(clean_system.conversation_memory.history)
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert str(result.get("status") or "").upper() in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE", "GENERATION_ABSTAIN"}
    after = len(clean_system.conversation_memory.history)
    assert after == before
    assert all("xylomediasis" not in question.casefold() for question, _ in clean_system.conversation_memory.history)


@pytest.mark.high_level
def test_conversation__production_blocked_outcome_is_not_persisted(clean_system, monkeypatch):
    before = list(clean_system.conversation_memory.history)

    def blocked_answer(*args, **kwargs):
        return {
            "status": "BLOCK",
            "answer": "This unsafe answer must never be stored.",
            "hits": [],
            "citations": [],
            "verification": {"allow": False, "blocked_claims": 1},
            "grounding": {"allow": False},
        }

    monkeypatch.setattr(clean_system, "_certified_god_answer", blocked_answer)
    result = clean_system.answer("Store nothing from this blocked request")

    assert str(result.get("status") or "").upper() == "BLOCK"
    assert clean_system.conversation_memory.history == before
    assert all("Store nothing from this blocked request" not in question for question, _ in clean_system.conversation_memory.history)


@pytest.mark.high_level
def test_conversation__non_success_structured_outcomes_are_rejected_by_memory_primitive(clean_system):
    memory = clean_system.conversation_memory
    before = list(memory.history)

    assert memory.add("blocked query", {"status": "BLOCK", "answer": "unsafe content"}) is False
    assert memory.add("abstained query", {"status": "GENERATION_ABSTAIN", "answer": "uncertain"}) is False
    assert memory.add("unsupported query", {"status": "NOT_SUPPORTED", "answer": "not enough evidence"}) is False
    assert memory.history == before
