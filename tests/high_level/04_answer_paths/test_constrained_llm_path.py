from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import (
    assert_citations_valid,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_llm_called_with_small_context,
    assert_pipeline_authority,
    assert_prompt_excludes,
)


@pytest.mark.high_level
def test_constrained_llm__uses_exact_path_with_grounded_context(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = (
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]\n"
        "Metformin is commonly used for type 2 diabetes. [S2]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Compare the role of HbA1c assessment with metformin treatment in type 2 diabetes "
        "and explain the treatment mechanism and clinical management implications."
    )

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    assert (result.get("generation_meta") or {}).get("attempted") is True
    assert_llm_called_with_small_context(fake_ollama_fast, max_tokens=700)
    assert_citations_valid(result)
    assert_grounded(result)
    assert_pipeline_authority(result)
    assert result.get("evidence_first") is True


@pytest.mark.high_level
def test_constrained_llm__passes_zero_temperature_to_llm(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]"
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and treatment management of diabetes mellitus using the indexed evidence."
    )

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    assert fake_ollama_fast.calls
    assert all(float((call.get("kwargs") or {}).get("temperature", -1)) == 0.0 for call in fake_ollama_fast.calls)
    assert_grounded(result)


@pytest.mark.high_level
def test_constrained_llm__context_only_from_retrieved_hits(clean_system, fake_ollama_fast, tmp_path):
    unrelated = write_minimal_pdf(
        tmp_path / "unrelated_indexed_note.pdf",
        ["UNRELATED_PRIVATE_FIXTURE_MARKER 8f42a19 veterinary dermatology note."],
    )
    ingestion = clean_system.ingest_file(unrelated)
    assert str(ingestion.get("status") or "").upper() == "READY"

    fake_ollama_fast.response = "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]"
    clean_system.llm = fake_ollama_fast
    result = clean_system.answer("Explain the mechanism and treatment of diabetes mellitus using the indexed evidence.")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    assert_llm_called_with_small_context(fake_ollama_fast, max_tokens=700)
    assert_prompt_excludes(fake_ollama_fast, ["UNRELATED_PRIVATE_FIXTURE_MARKER", "8f42a19"])
    assert_grounded(result)
    assert_citations_valid(result)


@pytest.mark.high_level
def test_constrained_llm__grounded_synthesis_contains_retrieved_source_marker(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = "The indexed evidence describes diabetes mellitus as a chronic metabolic disorder. [S1]"
    clean_system.llm = fake_ollama_fast
    result = clean_system.answer("Explain the mechanism and clinical significance of diabetes mellitus from the indexed evidence.")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    assert "[S1]" in str(result.get("answer") or "")
    assert_citations_valid(result)
    assert_grounded(result)
