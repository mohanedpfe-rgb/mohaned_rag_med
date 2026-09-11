import pytest

@pytest.mark.high_level

def test_multi_part_question__exposes_subqueries_or_plan(clean_system):
    result = clean_system.answer("What are the causes and complications of diabetes mellitus?")
    trace = result.get("query_trace") or {}
    analysis = result.get("query_analysis") or result.get("route") or {}
    assert any(key in str(trace) + str(analysis) for key in ("subquer", "decompos", "plan"))

@pytest.mark.high_level

def test_single_question__is_not_excessively_decomposed(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    trace = result.get("query_trace") or {}
    variants = str(trace).lower().count("subquery") + str(trace).lower().count("variant")
    assert variants < 50
