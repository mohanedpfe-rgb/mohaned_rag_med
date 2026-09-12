from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_llm_called_with_small_context, assert_status


@pytest.mark.high_level
def test_hard_question__selects_constrained_llm_path_with_grounded_context(clean_system, fake_ollama_fast):
    """A genuinely complex query must use the constrained LLM only after evidence retrieval."""
    fake_ollama_fast.response = (
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]\n"
        "Metformin is commonly used for type 2 diabetes. [S2]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Compare the role of HbA1c assessment with metformin treatment in type 2 diabetes "
        "and explain the treatment mechanism and clinical management implications."
    )

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert result.get("generation_path") == "PATH_C_CONSTRAINED_LLM", result
    assert (result.get("generation_meta") or {}).get("attempted") is True
    assert_llm_called_with_small_context(fake_ollama_fast, max_tokens=700)
    assert_citations_valid(result)
    assert_grounded(result)
    assert result.get("evidence_first") is True
    assert (result.get("query_trace") or {}).get("generation", {}).get("path") == "PATH_C_CONSTRAINED_LLM"


@pytest.mark.high_level
def test_hard_question__passes_zero_temperature_to_llm(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]"
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and treatment management of diabetes mellitus using the indexed evidence."
    )

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert fake_ollama_fast.calls, "complex query must make an LLM call"
    for call in fake_ollama_fast.calls:
        assert float((call.get("kwargs") or {}).get("temperature", -1)) == 0.0
    assert_grounded(result)


@pytest.mark.high_level
def test_hard_question__llm_prompt_contains_only_retrieved_evidence_not_unrelated_indexed_document(clean_system, fake_ollama_fast, tmp_path):
    """The synthesis model must receive bounded retrieved evidence, not unrelated indexed content."""
    unrelated = write_minimal_pdf(
        tmp_path / "unrelated_indexed_note.pdf",
        ["UNRELATED_PRIVATE_FIXTURE_MARKER 8f42a19 veterinary dermatology note."],
    )
    ingestion = clean_system.ingest_file(unrelated)
    assert str(ingestion.get("status") or "").upper() == "READY"

    fake_ollama_fast.response = "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]"
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain what diabetes mellitus is and how HbA1c is used for glycemic control."
    )

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert_llm_called_with_small_context(fake_ollama_fast, max_tokens=700)
    prompt = "\n".join(str(call.get("prompt") or "") for call in fake_ollama_fast.calls)
    assert "Evidence" in prompt or "evidence" in prompt
    assert "Diabetes mellitus is a chronic metabolic disorder" in prompt
    assert "UNRELATED_PRIVATE_FIXTURE_MARKER" not in prompt
    assert_grounded(result)
    assert_citations_valid(result)


@pytest.mark.high_level
def test_hard_question__grounded_llm_response_must_reference_retrieved_source_marker(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = "The indexed evidence describes diabetes mellitus as a chronic metabolic disorder characterized by hyperglycemia. [S1]"
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and clinical significance of diabetes mellitus from the indexed evidence."
    )

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert result.get("hits"), "LLM success must preserve retrieved evidence"
    assert "[S1]" in str(result.get("answer") or ""), result
    assert_citations_valid(result)
    assert_grounded(result)
