from __future__ import annotations

import pytest

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
