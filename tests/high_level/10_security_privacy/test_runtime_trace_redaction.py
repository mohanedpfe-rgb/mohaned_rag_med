from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_security__production_query_trace_redacts_email_phone_and_long_id_without_changing_answer_authority(clean_system):
    question = "What is diabetes mellitus? contact student@example.com phone +213 555 123 456 id 123456789012"
    result = clean_system.answer(question)

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    trace = result.get("query_trace") or {}
    serialized = str(trace)
    assert "student@example.com" not in serialized
    assert "+213 555 123 456" not in serialized
    assert "123456789012" not in serialized
    assert "[REDACTED_EMAIL]" in serialized
    assert "[REDACTED_PHONE]" in serialized
    assert "[REDACTED_ID]" in serialized
    assert_grounded(result)
