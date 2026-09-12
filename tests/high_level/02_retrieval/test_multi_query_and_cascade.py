from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_retrieval__high_confidence_simple_query_uses_exact_tier_zero_early_exit(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    retrieval = result.get("retrieval") or {}
    assert retrieval.get("early_exit") is True
    assert retrieval.get("tier") == "TIER0_EXIT"
    assert int(retrieval.get("candidate_count", 0)) >= 1
    assert int(retrieval.get("queries", 0)) == 1
    assert_grounded(result)
    assert_citations_valid(result)


@pytest.mark.high_level
def test_retrieval__complex_question_expands_queries_and_uses_exact_constrained_path(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = "The indexed evidence compares type 1 and type 2 diabetes mechanisms and treatment. [S1]"
    clean_system.llm = fake_ollama_fast
    result = clean_system.answer("Compare the mechanism, treatment, and contraindications of type 1 and type 2 diabetes.")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    retrieval = result.get("retrieval") or {}
    route = result.get("route") or {}
    variants = route.get("query_variants") or []
    assert retrieval.get("tier") in {"TIER1", "TIER2"}
    assert int(retrieval.get("candidate_count", 0)) >= 1
    assert int(retrieval.get("queries", 0)) >= 2
    assert len(variants) >= 2
    assert int(retrieval.get("queries", 0)) >= len(variants)
    assert_grounded(result)
    assert_citations_valid(result)
