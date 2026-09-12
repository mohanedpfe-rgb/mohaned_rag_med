from __future__ import annotations

import pytest

from tests.high_level.helpers import (
    assert_citations_valid,
    assert_grounded,
    assert_latency_under,
    assert_status,
    timed_call,
)


@pytest.mark.high_level
def test_simple_fact__completes_within_five_seconds_and_reports_trace_latency(clean_system):
    result, wall_clock_seconds = timed_call(clean_system.answer, "What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert str(result.get("answer") or "").strip()
    assert_latency_under(result, 5.0)

    reported_ms = float(result.get("latency_ms") or (result.get("query_trace") or {}).get("timings_ms", {}).get("total"))
    assert reported_ms >= 0.0
    assert wall_clock_seconds <= 5.25, f"wall-clock latency {wall_clock_seconds:.3f}s exceeded 5s budget plus measurement tolerance"
    assert reported_ms <= 5000.0

    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_numeric_query__stays_within_seven_seconds_and_preserves_latency_trace(clean_system):
    result, wall_clock_seconds = timed_call(
        clean_system.answer,
        "What numeric information is stated about HbA1c?",
    )

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED"})
    assert_latency_under(result, 7.0)
    assert wall_clock_seconds <= 7.25

    trace = result.get("query_trace") or {}
    timings = trace.get("timings_ms") or {}
    assert float(timings.get("total", 0.0)) >= 0.0
    assert float(timings.get("retrieval", 0.0)) >= 0.0
    assert float(timings.get("generation", 0.0)) >= 0.0

    if str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert_citations_valid(result)
        assert_grounded(result)
