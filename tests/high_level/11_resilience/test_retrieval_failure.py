from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_status


@pytest.mark.high_level
def test_resilience__retrieval_exception_returns_exact_answer_unavailable_without_crash(clean_system, monkeypatch):
    def fail_retrieve(*args, **kwargs):
        raise RuntimeError("simulated retrieval outage")

    monkeypatch.setattr(clean_system.retriever, "retrieve", fail_retrieve)
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "ANSWER_UNAVAILABLE")
    assert not result.get("generation_path")
    assert result.get("citations") == []
    recovery = result.get("recovery") or {}
    assert recovery.get("attempted") is True
    assert recovery.get("retrieval_failed") == "RuntimeError"
    assert recovery.get("pipeline_error") in {"RuntimeError", "Exception"}
    assert "Traceback" not in str(result.get("answer") or "")
