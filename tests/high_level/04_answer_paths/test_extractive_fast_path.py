from __future__ import annotations

import pytest

from tests.high_level.helpers import (
    assert_citations_valid,
    assert_grounded,
    assert_latency_under,
    assert_status,
)


@pytest.mark.high_level
def test_simple_fact__uses_extractive_path_without_llm(clean_system, fake_ollama_fast):
    """A simple factual question must stay on the deterministic extraction path."""
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert result.get("generation_path") == "PATH_A_EXTRACTIVE", result
    assert (result.get("generation_meta") or {}).get("attempted") is False
    assert fake_ollama_fast.calls == []
    assert_grounded(result)
    assert_citations_valid(result)
    assert result.get("evidence_first") is True
    assert result.get("canonical_pipeline_executed") is True
    assert_latency_under(result, 5.0)


@pytest.mark.high_level
def test_simple_fact__proves_citations_reference_retrieved_hits(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    hits = result.get("hits") or []
    answer = str(result.get("answer") or "")
    assert hits, "successful simple answer must have retrieved evidence"
    assert "[S1]" in answer, f"answer has no first-source citation: {answer!r}"
    assert_citations_valid(result)
    assert_grounded(result)
