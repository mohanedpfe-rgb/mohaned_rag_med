from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_simple_answer__declares_exact_selected_generation_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    plan = result.get("answer_plan") or {}
    assert plan.get("selected_path") == result.get("generation_path")
    assert result.get("generation_path") == "PATH_A_EXTRACTIVE"
    assert (result.get("phases") or {}).get("phase_4_answer_cascade") == "PATH_A_EXTRACTIVE"


@pytest.mark.high_level
def test_comparison_answer__uses_template_or_constrained_path_and_remains_grounded(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = (
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]\n"
        "Metformin is commonly used for type 2 diabetes. [S2]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Compare diabetes mellitus and type 2 diabetes management, including treatment and mechanism, "
        "using only the indexed evidence."
    )

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert result.get("generation_path") in {
        "PATH_B_TEMPLATE",
        "PATH_C_CONSTRAINED_LLM",
        "PATH_HYBRID_FALLBACK",
        "PATH_A_VERIFIED_FALLBACK",
    }, result
    assert (result.get("answer_plan") or {}).get("selected_path") == result.get("generation_path")
    assert_citations_valid(result)
    assert_grounded(result)
