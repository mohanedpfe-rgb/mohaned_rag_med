from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_resilience__ollama_failure_never_turns_into_uncited_success(clean_system, fake_ollama_fast):
    def fail_generate(*args, **kwargs):
        raise RuntimeError("simulated Ollama outage")

    fake_ollama_fast.generate = fail_generate
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management of diabetes mellitus using the indexed evidence."
    )

    status = str(result.get("status") or "").upper()
    assert status in {"SUCCESS_WITH_WARNINGS", "ANSWER_UNAVAILABLE", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}
    if status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert result.get("hits")
        assert result.get("citations")
        assert_grounded(result)
        assert_citations_valid(result)
    else:
        assert not result.get("citations")

    recovery = result.get("recovery") or {}
    if recovery:
        assert recovery.get("pipeline_error") in {"RuntimeError", None}


@pytest.mark.high_level
def test_resilience__successful_requests_retain_runtime_safety_observability(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})

    safety = result.get("runtime_safety") or {}
    assert safety.get("ready_evidence_enforced") is True
    assert safety.get("success_contract_enforced") is True
