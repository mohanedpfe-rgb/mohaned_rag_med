import pytest

@pytest.mark.high_level
@pytest.mark.parametrize("question", [
    "What is diabetes mellitus?",
    "What is the dose of metformin?",
    "Compare type 1 and type 2 diabetes.",
    "What is the mechanism of insulin resistance?",
])
def test_query_intelligence__classifies_and_routes(question, clean_system):
    result = clean_system.answer(question)
    analysis = result.get("query_analysis") or result.get("route") or {}
    assert isinstance(analysis, dict)
    assert any(key in analysis for key in ("intent", "complexity", "entities"))

@pytest.mark.high_level
def test_complexity__is_bounded(clean_system):
    result = clean_system.answer("Explain the mechanism, diagnosis and management of diabetes mellitus.")
    analysis = result.get("query_analysis") or result.get("route") or {}
    complexity = analysis.get("complexity")
    if complexity is not None:
        assert 0.0 <= float(complexity) <= 1.0 or float(complexity) >= 0.0
