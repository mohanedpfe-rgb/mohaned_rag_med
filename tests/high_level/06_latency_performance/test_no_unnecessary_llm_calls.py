from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_grounded, assert_status


@pytest.mark.high_level
def test_simple_factual_answer__uses_extractive_path_and_makes_zero_llm_calls(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert result.get("generation_path") == "PATH_A_EXTRACTIVE", result
    assert (result.get("generation_meta") or {}).get("attempted") is False
    assert fake_ollama_fast.calls == []
    assert result.get("canonical_pipeline_executed") is True
    assert result.get("evidence_first") is True
    assert_grounded(result)


@pytest.mark.high_level
def test_simple_factual_answer__does_not_invoke_llm_for_latency_optimization(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer("Define diabetes mellitus using the indexed evidence.")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert fake_ollama_fast.calls == []
    assert result.get("generation_path") in {"PATH_A_EXTRACTIVE", "PATH_A_VERIFIED_FALLBACK"}, result


@pytest.mark.high_level
def test_runtime__keeps_ollama_concurrency_within_small_local_machine_bound(clean_system):
    concurrency = int(clean_system.settings.ollama_concurrency)
    assert 1 <= concurrency <= 2
