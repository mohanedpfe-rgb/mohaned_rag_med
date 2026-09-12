from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded, assert_latency_under, assert_pipeline_authority, timed_call


@pytest.mark.high_level
@pytest.mark.slow
def test_complex_question__stays_under_configured_generation_ceiling(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = (
        "Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia. [S1]\n"
        "Metformin is commonly used for type 2 diabetes. [S2]"
    )
    clean_system.llm = fake_ollama_fast

    result, wall_clock_seconds = timed_call(
        clean_system.answer,
        "Compare the causes, mechanisms, diagnosis and management of type 1 and type 2 diabetes mellitus, "
        "using only the indexed evidence.",
    )

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_C_CONSTRAINED_LLM")
    assert_pipeline_authority(result)
    assert fake_ollama_fast.calls, "complex synthesis query must invoke the LLM"

    configured_budget_seconds = float(clean_system.settings.generation_latency_budget_seconds)
    assert_latency_under(result, configured_budget_seconds)
    assert wall_clock_seconds <= configured_budget_seconds + 0.75

    trace = result.get("query_trace") or {}
    timings = trace.get("timings_ms") or {}
    total_ms = float(timings.get("total", 0.0))
    assert total_ms > 0.0
    assert total_ms <= configured_budget_seconds * 1000.0

    assert (result.get("generation_meta") or {}).get("attempted") is True
    assert (result.get("query_trace") or {}).get("generation", {}).get("path") == "PATH_C_CONSTRAINED_LLM"
    assert_citations_valid(result)
    assert_grounded(result)
