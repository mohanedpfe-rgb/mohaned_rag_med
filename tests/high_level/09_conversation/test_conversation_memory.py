from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded, assert_memory_unchanged


@pytest.mark.high_level
def test_conversation__successful_answer_supports_exact_metformin_followup_path(clean_system):
    first = clean_system.answer("What is metformin used for in type 2 diabetes?")
    assert_exact_status(first, "SUCCESS")
    assert_exact_path(first, "PATH_A_EXTRACTIVE")
    assert_grounded(first)
    assert_citations_valid(first)

    before_followup = len(clean_system.conversation_memory.history)
    second = clean_system.answer("What about its dose?")
    assert_exact_status(second, "SUCCESS")
    assert_exact_path(second, "PATH_B_TEMPLATE")
    route = second.get("route") or {}
    assert route.get("is_follow_up") is True
    rewritten = str(second.get("rewritten_question") or "").casefold()
    assert "metformin" in rewritten
    assert "dose" in rewritten
    assert "500 mg" in str(second.get("answer") or "")
    assert_citations_valid(second)
    assert_grounded(second)
    assert len(clean_system.conversation_memory.history) == before_followup + 1
    assert sum("what about its dose" in question.casefold() for question, _ in clean_system.conversation_memory.history) == 1


@pytest.mark.high_level
def test_conversation__failed_answer_is_not_persisted(clean_system):
    before = list(clean_system.conversation_memory.history)
    result = clean_system.answer("What is the cure for a fictional disease called xylomediasis?")

    assert_exact_status(result, "NOT_SUPPORTED")
    assert_memory_unchanged(before, clean_system.conversation_memory.history)
    assert all("xylomediasis" not in question.casefold() for question, _ in clean_system.conversation_memory.history)


@pytest.mark.high_level
def test_conversation__blocked_production_outcome_is_not_persisted(clean_system, monkeypatch):
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
    assert_memory_unchanged(before, clean_system.conversation_memory.history)


@pytest.mark.high_level
def test_conversation__memory_primitive_accepts_success_only(clean_system):
    memory = clean_system.conversation_memory
    before = list(memory.history)

    assert memory.add("blocked query", {"status": "BLOCK", "answer": "unsafe content"}) is False
    assert memory.add("abstained query", {"status": "GENERATION_ABSTAIN", "answer": "uncertain"}) is False
    assert memory.add("unsupported query", {"status": "NOT_SUPPORTED", "answer": "not enough evidence"}) is False
    assert memory.history == before
