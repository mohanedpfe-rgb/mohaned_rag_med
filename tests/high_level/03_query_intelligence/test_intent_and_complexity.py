from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_entities_present, assert_exact_path, assert_exact_status


@pytest.mark.high_level
def test_query_intelligence__classifies_numeric_medical_question_and_selects_exact_template_path(clean_system):
    result = clean_system.answer("What dose of metformin is explicitly stated for type 2 diabetes?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    route = result.get("route") or {}
    assert route.get("intent") == "numeric"
    assert route.get("numeric_sensitivity") is True
    assert float(route.get("confidence_threshold", 0)) >= 0.75
    assert_entities_present(result, {"metformin"})


@pytest.mark.high_level
def test_query_intelligence__classifies_comparison_as_hard_and_selects_exact_constrained_llm_path(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = "The indexed evidence compares the diabetes mechanisms and treatment options. [S1]"
    clean_system.llm = fake_ollama_fast
    result = clean_system.answer("Compare type 1 and type 2 diabetes mechanisms, treatment, and contraindications.")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    route = result.get("route") or {}
    assert route.get("intent") == "comparison"
    assert float(route.get("complexity", 0)) >= 0.25
    assert route.get("needs_multi_hop") is True or float(route.get("complexity", 0)) >= 0.60


@pytest.mark.high_level
def test_query_intelligence__exposes_real_entities_for_relationship_question(clean_system):
    result = clean_system.answer("What is the relationship between HbA1c and glycemic control in diabetes?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    entities = [str(item).casefold() for item in (result.get("route") or {}).get("entities", [])]
    assert any("hba1c" in entity for entity in entities)
    assert any("diabetes" in entity for entity in entities)
    assert all(entity not in {"and", "the", "what", "is", "between"} for entity in entities)
