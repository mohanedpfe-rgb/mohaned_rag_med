import pytest
from tests.high_level.helpers import assert_latency_under

@pytest.mark.high_level
@pytest.mark.slow

def test_complex_question__never_exceeds_hard_ceiling(clean_system):
    result = clean_system.answer("Compare causes, mechanisms, diagnosis and management of type 1 and type 2 diabetes mellitus.")
    assert_latency_under(result, max(15.0, float(clean_system.settings.generation_latency_budget_seconds) + 2.0))
