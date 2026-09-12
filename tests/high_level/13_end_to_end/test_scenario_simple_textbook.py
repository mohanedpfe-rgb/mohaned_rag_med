from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import (
    assert_citations_valid,
    assert_document_ready,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_latency_under,
    assert_pipeline_authority,
)


@pytest.mark.high_level
def test_e2e_textbook__ingest_ready_then_five_factual_questions_are_exact_extractive_successes(clean_system, tmp_path):
    textbook = write_minimal_pdf(tmp_path / "textbook.pdf", [
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia.",
        "HbA1c is used to assess glycemic control. Metformin is commonly used for type 2 diabetes.",
        "The controlled treatment dose in this fixture is 500 mg twice daily. Table 1: HbA1c target 7%.",
    ])
    ingestion = clean_system.ingest_file(textbook)
    assert str(ingestion.get("status") or "").upper() == "READY"
    document_id = str(ingestion.get("document_id") or ingestion.get("id") or "")
    assert_document_ready(clean_system, document_id)

    questions = [
        "What is diabetes mellitus?",
        "What characterizes diabetes mellitus?",
        "What is metformin used for?",
        "What is HbA1c used to assess?",
        "What does Table 1 state about HbA1c?",
    ]
    before = len(getattr(clean_system.conversation_memory, "history", []) or [])
    for question in questions:
        result = clean_system.answer(question)
        assert_exact_status(result, "SUCCESS")
        assert_exact_path(result, "PATH_A_EXTRACTIVE")
        assert result.get("hits")
        assert_citations_valid(result)
        assert_grounded(result)
        assert_pipeline_authority(result)
        assert_latency_under(result, 5.0)
        assert result.get("evidence_first") is True
    after = len(getattr(clean_system.conversation_memory, "history", []) or [])
    assert after == before + len(questions)
