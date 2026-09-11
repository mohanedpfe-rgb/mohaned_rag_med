import pytest
from tests.high_level.helpers import assert_latency_under

@pytest.mark.high_level

def test_simple_fact__has_measurable_latency(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_latency_under(result, 5.0)

@pytest.mark.high_level

def test_numeric_query__has_measurable_latency(clean_system):
    result = clean_system.answer("What numeric information is stated about HbA1c?")
    assert_latency_under(result, 7.0)
