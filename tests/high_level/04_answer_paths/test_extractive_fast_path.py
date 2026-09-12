from __future__ import annotations

import pytest

from tests.high_level.helpers import (
    assert_citations_valid,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_latency_under,
)


@pytest.mark.high_level
def test_extractive__never_calls_llm_and_path_is_exact(clean_system, fake_ollama_fast):
    clean_system.llm = fake_ollama_fast
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert (result.get("generation_meta") or {}).get("attempted") is False
    assert fake_ollama_fast.calls == []
    assert_latency_under(result, 5.0)
    assert_grounded(result)
    assert_citations_valid(result)


@pytest.mark.high_level
def test_extractive__citations_resolve_to_retrieved_hits(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert result.get("hits")
    assert "[S1]" in str(result.get("answer") or "")
    assert_citations_valid(result)
    assert_grounded(result)
