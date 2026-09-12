from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_resilience__ollama_outage_recovers_to_verified_extractive_answer(clean_system, fake_ollama_fast):
    def fail_generate(*args, **kwargs):
        raise RuntimeError("simulated Ollama outage")

    fake_ollama_fast.generate = fail_generate
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management of diabetes mellitus using the indexed evidence."
    )

    assert_status(result, {"SUCCESS_WITH_WARNINGS", "ANSWER_UNAVAILABLE", "NOT_SUPPORTED"})
    recovery = result.get("recovery") or {}
    assert recovery.get("attempted") is True
    assert recovery.get("pipeline_error") == "RuntimeError"
    assert "Traceback" not in str(result.get("answer") or "")

    if str(result.get("status") or "").upper() == "SUCCESS_WITH_WARNINGS":
        assert recovery.get("grounded_extractive_fallback") is True
        assert result.get("generation_path") == "PATH_A_VERIFIED_FALLBACK"
        assert result.get("hits")
        assert_citations_valid(result)
        assert_grounded(result)
        assert result.get("phase_implementation", {}).get("degraded_to_recovery") is True
        assert (result.get("query_trace") or {}).get("generation", {}).get("status") == "extractive"
