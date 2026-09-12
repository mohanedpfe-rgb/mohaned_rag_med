from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_status


@pytest.mark.high_level
def test_query_intelligence__multi_part_question_is_split_into_cause_and_complication_variants(clean_system):
    result = clean_system.answer("What are the causes and complications of diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    route = result.get("route") or {}
    variants = [str(item).casefold() for item in (route.get("query_variants") or [])]
    assert len(variants) >= 2
    combined = " | ".join(variants)
    assert "cause" in combined or "causes" in combined
    assert "complication" in combined or "complications" in combined
    assert float(route.get("complexity", 0)) >= 0.20
    assert route.get("needs_multi_hop") is True or len(variants) >= 2


@pytest.mark.high_level
def test_query_intelligence__single_factual_question_is_not_over_decomposed(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    route = result.get("route") or {}
    variants = route.get("query_variants") or []
    assert len(variants) <= 2
    assert route.get("needs_multi_hop") is False
    assert float(route.get("complexity", 0)) < 0.65
