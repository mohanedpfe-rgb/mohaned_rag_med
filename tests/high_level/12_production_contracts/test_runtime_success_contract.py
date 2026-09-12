from __future__ import annotations

import pytest

from rag_project.intelligence.runtime_safety import execute_with_runtime_safety


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

    assert result.get("status") == "GENERATION_ABSTAIN"
    assert result.get("citations") == []
    assert result.get("needs_review") is True
    assert (result.get("runtime_safety") or {}).get("success_contract_enforced") is True


@pytest.mark.high_level
def test_runtime_success_contract__success_with_verified_evidence_remains_success(clean_system):
    first = clean_system.answer("What is diabetes mellitus?")
    assert first.get("status") in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    assert first.get("hits")
    assert (first.get("grounding") or {}).get("allow") is True
    assert first.get("citations") or "[S" in str(first.get("answer") or "")
    assert (first.get("runtime_safety") or {}).get("success_contract_enforced") is True
