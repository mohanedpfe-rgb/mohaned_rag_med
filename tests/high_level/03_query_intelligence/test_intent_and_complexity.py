from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_status, assert_entities_present


@pytest.mark.high_level
def test_query_intelligence__classifies_numeric_medical_question_correctly(clean_system):
    result = clean_system.answer("What dose of metformin is explicitly stated for type 2 diabetes?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    route = result.get("route") or {}
    assert route.get("intent") == "numeric"
    assert route.get("numeric_sensitivity") is True
    assert float(route.get("confidence_threshold", 0)) >= 0.75
    assert_entities_present(result, {"metformin"})


@pytest.mark.high_level
def test_query_intelligence__classifies_comparison_and_raises_complexity(clean_system):
    result = clean_system.answer("Compare type 1 and type 2 diabetes mechanisms, treatment, and contraindications.")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "GENERATION_ABSTAIN"})
    route = result.get("route") or {}
    assert route.get("intent") == "comparison"
    assert float(route.get("complexity", 0)) >= 0.25
    assert route.get("needs_multi_hop") is True or float(route.get("complexity", 0)) >= 0.60


@pytest.mark.high_level
def test_query_intelligence__exposes_real_entities_not_function_words(clean_system):
    result = clean_system.answer("What is the relationship between HbA1c and glycemic control in diabetes?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED"})
    entities = [str(item).casefold() for item in (result.get("route") or {}).get("entities", [])]
    assert any("hba1c" in entity for entity in entities)
    assert any("diabetes" in entity for entity in entities)
    assert all(entity not in {"and", "the", "what", "is", "between"} for entity in entities)
