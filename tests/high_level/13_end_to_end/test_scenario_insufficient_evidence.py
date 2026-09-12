from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_abstained


@pytest.mark.high_level
def test_e2e_abstain__weak_or_absent_evidence_is_not_answered_as_fact(clean_system):
    result = clean_system.answer("What is the definitive cure and exact molecular mechanism of xylomediasis?")

    assert_abstained(result)
    assert str(result.get("status") or "").upper() in {"NOT_SUPPORTED", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE", "ABSTAIN", "BLOCK"}
    assert not (result.get("claims") or [])
