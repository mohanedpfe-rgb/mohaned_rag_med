from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status, assert_latency_under, assert_pipeline_authority, timed_call


@pytest.mark.high_level
def test_latency__successful_simple_answer_exposes_total_latency_inside_five_second_budget(clean_system):
    result, wall = timed_call(clean_system.answer, "What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert_latency_under(result, 5.0)
    assert wall <= 5.0

    trace = result.get("query_trace") or {}
    timings = trace.get("timings_ms") or {}
    assert float(timings.get("total", -1.0)) >= 0.0
    assert float(timings.get("total", 999999.0)) <= 5000.0
    assert_pipeline_authority(result)


@pytest.mark.high_level
def test_latency__simple_answer_does_not_pay_generation_budget_when_llm_is_available(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast
    result, wall = timed_call(clean_system.answer, "What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert fake_ollama_fast.calls == []
    assert wall <= 5.0
