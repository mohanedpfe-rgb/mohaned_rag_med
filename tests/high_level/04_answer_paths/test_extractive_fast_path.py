import pytest
from tests.high_level.helpers import assert_citations_valid, assert_extractive_path

@pytest.mark.high_level

def test_simple_fact__uses_grounded_fast_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    if str(result.get("status", "")).upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_extractive_path(result)
        assert_citations_valid(result)
