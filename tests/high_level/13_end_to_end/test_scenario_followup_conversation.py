from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_e2e_followup__four_successful_turns_preserve_context_then_isolate_new_topic(clean_system, tmp_path):
    hypertension = write_minimal_pdf(tmp_path / "hypertension.pdf", [
        "Hypertension is persistent high blood pressure. The controlled target in this fixture is below 140/90 mmHg."
    ])
    assert str(clean_system.ingest_file(hypertension).get("status") or "").upper() == "READY"

    before = len(clean_system.conversation_memory.history)

    first = clean_system.answer("What is diabetes mellitus?")
    assert_exact_status(first, "SUCCESS")
    assert_exact_path(first, "PATH_A_EXTRACTIVE")
    assert_grounded(first)
    assert_citations_valid(first)

    followup = clean_system.answer("What about HbA1c?")
    assert_exact_status(followup, "SUCCESS")
    assert_exact_path(followup, "PATH_A_EXTRACTIVE")
    assert (followup.get("route") or {}).get("is_follow_up") is True
    rewritten = str(followup.get("rewritten_question") or "").casefold()
    assert "diabetes" in rewritten
    assert "hba1c" in rewritten
    assert_grounded(followup)
    assert_citations_valid(followup)

    switched = clean_system.answer("What is hypertension?")
    assert_exact_status(switched, "SUCCESS")
    assert_exact_path(switched, "PATH_A_EXTRACTIVE")
    switched_text = str(switched.get("answer") or "").casefold()
    assert "hypertension" in switched_text
    assert_grounded(switched)
    assert_citations_valid(switched)

    hypertension_followup = clean_system.answer("What is its controlled target?")
    assert_exact_status(hypertension_followup, "SUCCESS")
    assert_exact_path(hypertension_followup, "PATH_B_TEMPLATE")
    assert (hypertension_followup.get("route") or {}).get("is_follow_up") is True
    rewritten_h = str(hypertension_followup.get("rewritten_question") or "").casefold()
    assert "hypertension" in rewritten_h
    assert "target" in rewritten_h
    assert "140/90" in str(hypertension_followup.get("answer") or "")
    assert_grounded(hypertension_followup)
    assert_citations_valid(hypertension_followup)

    assert len(clean_system.conversation_memory.history) == before + 4
    stored_questions = [question.casefold() for question, _ in clean_system.conversation_memory.history[-4:]]
    assert stored_questions == [
        "what is diabetes mellitus?",
        "what about hba1c?",
        "what is hypertension?",
        "what is its controlled target?",
    ]
