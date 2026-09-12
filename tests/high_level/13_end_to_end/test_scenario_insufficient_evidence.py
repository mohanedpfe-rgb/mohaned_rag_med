from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_status, assert_memory_unchanged


@pytest.mark.high_level
def test_e2e_abstain__weak_or_absent_evidence_returns_exact_not_supported_without_memory_write(clean_system):
    before = list(clean_system.conversation_memory.history)
    result = clean_system.answer("What is the definitive cure and exact molecular mechanism of xylomediasis?")

    assert_exact_status(result, "NOT_SUPPORTED")
    assert not result.get("generation_path")
    assert not result.get("claims")
    assert not result.get("hits")
    assert not result.get("citations")
    assert_memory_unchanged(before, clean_system.conversation_memory.history)
