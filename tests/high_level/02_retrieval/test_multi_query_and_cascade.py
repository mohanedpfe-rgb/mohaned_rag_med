from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_retrieval__high_confidence_simple_query_uses_early_exit(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    retrieval = result.get("retrieval") or {}
    assert retrieval.get("early_exit") is True
    assert retrieval.get("tier") in {"CACHE", "TIER0_EXIT", "TIER1"}
    assert int(retrieval.get("candidate_count", 0)) >= 1
    assert int(retrieval.get("queries", 0)) == 1
    assert_grounded(result)
    assert_citations_valid(result)


@pytest.mark.high_level
def test_retrieval__complex_question_exposes_query_expansion_and_nontrivial_tier(clean_system):
    result = clean_system.answer("Compare the mechanism, treatment, and contraindications of type 1 and type 2 diabetes.")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "GENERATION_ABSTAIN", "NOT_SUPPORTED"})
    retrieval = result.get("retrieval") or {}
    route = result.get("route") or {}
    variants = route.get("query_variants") or []

    assert retrieval.get("tier") in {"TIER1", "TIER2", "CACHE", "TIER0_EXIT"}
    assert int(retrieval.get("candidate_count", 0)) >= 1
    assert int(retrieval.get("queries", 0)) >= 1
    assert len(variants) >= 2 or route.get("needs_multi_hop") is True
    assert int(retrieval.get("queries", 0)) >= max(1, len(variants))

    if str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_grounded(result)
        assert_citations_valid(result)
