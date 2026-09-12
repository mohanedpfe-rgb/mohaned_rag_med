from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_latency__simple_factual_answer_uses_exact_extractive_path_with_zero_llm_calls(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert (result.get("generation_meta") or {}).get("attempted") is False
    assert fake_ollama_fast.calls == []
    assert result.get("canonical_pipeline_executed") is True
    assert result.get("evidence_first") is True
    assert_grounded(result)


@pytest.mark.high_level
def test_latency__simple_definition_does_not_invoke_llm_for_exact_extractive_path(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast
    result = clean_system.answer("Define diabetes mellitus using the indexed evidence.")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert fake_ollama_fast.calls == []


@pytest.mark.high_level
def test_latency__local_ollama_concurrency_is_capped_at_two_for_small_machine_runtime(clean_system):
    concurrency = int(clean_system.settings.ollama_concurrency)
    assert 1 <= concurrency <= 2
