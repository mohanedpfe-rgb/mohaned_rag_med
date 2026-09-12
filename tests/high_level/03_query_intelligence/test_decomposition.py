from __future__ import annotations

import pytest


@pytest.mark.high_level
def test_query_intelligence__multi_part_question_exposes_multiple_search_variants(clean_system):
    result = clean_system.answer("What are the causes and complications of diabetes mellitus?")

    assert str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "GENERATION_ABSTAIN", "NOT_SUPPORTED"}
    route = result.get("route") or {}
    variants = route.get("query_variants") or []
    assert len(variants) >= 2 or route.get("needs_multi_hop") is True
    assert float(route.get("complexity", 0)) >= 0.20


@pytest.mark.high_level
def test_query_intelligence__single_question_is_not_over_decomposed(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    route = result.get("route") or {}
    variants = route.get("query_variants") or []
    assert len(variants) <= 8
    assert route.get("needs_multi_hop") is False or float(route.get("complexity", 0)) >= 0.65
