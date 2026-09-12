from __future__ import annotations

import pytest

from rag_project.intelligence.runtime_safety import execute_with_runtime_safety
from tests.high_level.helpers import assert_exact_path, assert_exact_status, assert_pipeline_authority


@pytest.mark.high_level
def test_runtime_success_contract__success_without_grounding_or_citations_is_withheld(clean_system):
    def malformed_answer():
        return {
            "status": "SUCCESS",
            "answer": "Unsupported answer without evidence.",
            "hits": [],
            "citations": [],
            "verification": {"allow": False},
            "grounding": {"allow": False},
        }

    result = execute_with_runtime_safety(clean_system, "malformed success", malformed_answer)

    assert_exact_status(result, "GENERATION_ABSTAIN")
    assert not result.get("generation_path")
    assert result.get("citations") == []
    assert result.get("needs_review") is True
    assert (result.get("runtime_safety") or {}).get("success_contract_enforced") is True


@pytest.mark.high_level
def test_runtime_success_contract__success_with_verified_evidence_remains_exact_success(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert_pipeline_authority(result)
    assert result.get("hits")
    assert (result.get("verification") or {}).get("allow") is True
    assert (result.get("grounding") or {}).get("allow") is True
    assert float((result.get("verification") or {}).get("supported_ratio", 0.0)) >= 0.70
    assert result.get("citations") or "[S" in str(result.get("answer") or "")
    assert (result.get("runtime_safety") or {}).get("success_contract_enforced") is True
