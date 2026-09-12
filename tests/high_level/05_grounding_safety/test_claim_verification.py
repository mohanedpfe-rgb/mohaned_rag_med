from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_abstained, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_grounding__supported_answer_is_verified_above_seventy_percent_with_exact_success_path(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    verification = result.get("verification") or {}
    assert verification.get("checked") is True
    assert verification.get("allow") is True
    assert float(verification.get("supported_ratio", 0.0)) >= 0.70
    assert int(verification.get("blocked_claims", 0)) == 0
    assert_grounded(result)


@pytest.mark.high_level
def test_grounding__unsupported_llm_claim_is_exactly_rejected_without_surviving_claims(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = (
        "Metformin cures every form of cancer and eliminates diabetes permanently. [S1]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the mechanism and management implications of diabetes mellitus using only the indexed evidence."
    )

    assert_exact_status(result, "GENERATION_ABSTAIN")
    assert not result.get("generation_path") or str(result.get("generation_path")).startswith("PATH_C")
    assert_abstained(result)
    answer = str(result.get("answer") or "").casefold()
    assert "cures every form of cancer" not in answer
    assert "eliminates diabetes permanently" not in answer
