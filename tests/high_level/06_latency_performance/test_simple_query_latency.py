from __future__ import annotations

import pytest

from tests.high_level.helpers import (
    assert_citations_valid,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_latency_under,
    assert_no_llm_called,
    timed_call,
)


@pytest.mark.high_level
def test_simple_fact__wall_clock_and_reported_latency_are_both_under_five_seconds(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast
    result, wall_clock_seconds = timed_call(clean_system.answer, "What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert (result.get("generation_meta") or {}).get("attempted") is False
    assert_no_llm_called(fake_ollama_fast)
    assert_latency_under(result, 5.0)

    reported_ms = float(result.get("latency_ms") or (result.get("query_trace") or {}).get("timings_ms", {}).get("total"))
    assert 0.0 <= reported_ms <= 5000.0
    assert wall_clock_seconds <= 5.0
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_numeric_query__stays_within_seven_seconds_with_exact_template_path(clean_system):
    result, wall_clock_seconds = timed_call(
        clean_system.answer,
        "What numeric information is stated about HbA1c?",
    )

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    assert_latency_under(result, 7.0)
    assert wall_clock_seconds <= 7.0

    trace = result.get("query_trace") or {}
    timings = trace.get("timings_ms") or {}
    assert float(timings.get("total", 0.0)) > 0.0
    assert float(timings.get("retrieval", 0.0)) >= 0.0
    assert float(timings.get("generation", 0.0)) >= 0.0
    assert_citations_valid(result)
    assert_grounded(result)
