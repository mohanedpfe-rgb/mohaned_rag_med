import pytest

@pytest.mark.high_level

def test_simple_question__does_not_over_expand(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    trace = result.get("query_trace") or {}
    routing = trace.get("routing") or result.get("query_analysis") or {}
    variants = routing.get("query_variants") or routing.get("variants") or []
    assert len(variants) <= int(getattr(clean_system.settings, "max_query_variants", 8))

@pytest.mark.high_level

def test_complex_question__has_bounded_retrieval_budget(clean_system):
    result = clean_system.answer("Compare the mechanism, causes, diagnosis and management of diabetes mellitus and explain the important differences.")
    retrieval = result.get("retrieval") or {}
    assert isinstance(retrieval, dict)
    assert int(getattr(clean_system.settings, "max_query_variants", 8)) <= 12
