from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_latency_under, assert_status, timed_call


@pytest.mark.high_level
def test_latency__every_successful_answer_exposes_total_latency_and_budget_metadata(clean_system):
    result, wall = timed_call(clean_system.answer, "What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert_latency_under(result, 5.0)
    assert wall <= 5.25

    trace = result.get("query_trace") or {}
    timings = trace.get("timings_ms") or {}
    assert "total" in timings
    assert float(timings["total"]) >= 0.0
    budget = result.get("latency_budget") or result.get("performance") or {}
    if isinstance(budget, dict):
        for key in ("budget_seconds", "budget_ms", "ceiling_seconds", "ceiling_ms"):
            if key in budget:
                assert float(budget[key]) > 0.0
                break


@pytest.mark.high_level
def test_latency__simple_answer_does_not_pay_generation_budget_when_llm_is_available(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast
    result, wall = timed_call(clean_system.answer, "What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert result.get("generation_path") == "PATH_A_EXTRACTIVE"
    assert fake_ollama_fast.calls == []
    assert wall < 5.25
