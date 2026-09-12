from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_grounded, assert_status


@pytest.mark.high_level
def test_grounding__supported_answer_is_explicitly_verified(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    verification = result.get("verification") or {}
    assert verification.get("checked") is True
    assert verification.get("allow") is True
    assert float(verification.get("supported_ratio", 0.0)) >= 0.70
    assert int(verification.get("blocked_claims", 0)) == 0
    assert_grounded(result)


@pytest.mark.high_level
def test_grounding__unsupported_llm_claim_is_rejected_or_removed(clean_system, fake_ollama_fast):
    fake_ollama_fast.response = (
        "Metformin cures every form of cancer and eliminates diabetes permanently. [S1]"
    )
    clean_system.llm = fake_ollama_fast

    result = clean_system.answer(
        "Explain the management of diabetes mellitus and the role of metformin using only the indexed evidence."
    )

    answer = str(result.get("answer") or "").casefold()
    assert "cures every form of cancer" not in answer
    assert "eliminates diabetes permanently" not in answer
    assert result.get("status") in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "GENERATION_ABSTAIN"}
    if str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
        assert (result.get("verification") or {}).get("allow") is True
