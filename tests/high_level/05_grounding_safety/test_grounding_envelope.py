from __future__ import annotations

import pytest

from rag_project.intelligence.runtime_safety import execute_with_runtime_safety
from tests.high_level.helpers import assert_abstained, assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_grounding__successful_answer_requires_exact_extractive_path_and_source_attribution(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert result.get("hits")
    assert_citations_valid(result)
    assert_grounded(result)
    assert "[S1]" in str(result.get("answer") or "") or result.get("citations")


@pytest.mark.high_level
def test_grounding__malformed_success_is_converted_to_exact_generation_abstain(clean_system):
    def malformed():
        return {
            "status": "SUCCESS",
            "answer": "This medical claim is unsupported and uncited.",
            "hits": [],
            "citations": [],
            "verification": {"allow": True},
            "grounding": {"allow": True},
        }

    result = execute_with_runtime_safety(clean_system, "unsupported claim", malformed)

    assert_exact_status(result, "GENERATION_ABSTAIN")
    assert not result.get("generation_path")
    assert_abstained(result)
    assert result.get("citations") == []
    safety = result.get("runtime_safety") or {}
    assert safety.get("success_contract_enforced") is True
    assert safety.get("rejected_success_contract") is True
