from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_simple_answer__declares_exact_extractive_generation_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert (result.get("phases") or {}).get("phase_4_answer_cascade") == "PATH_A_EXTRACTIVE"
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_comparison_answer__uses_exact_constrained_llm_path_and_stays_grounded(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = (
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]\n"
        "Metformin is commonly used for type 2 diabetes. [S2]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Compare diabetes mellitus and type 2 diabetes management, including treatment and mechanism, "
        "using only the indexed evidence."
    )

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    assert fake_ollama_fast.calls
    assert_citations_valid(result)
    assert_grounded(result)
