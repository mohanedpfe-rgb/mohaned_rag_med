from __future__ import annotations

import pytest

from rag_project.intelligence.production_contract import redact_sensitive_text, sanitize_trace


@pytest.mark.high_level
def test_security__diagnostic_text_redacts_email_phone_and_long_identifiers():
    raw = "contact student@example.com phone +213 555 123 456 id 123456789012"
    redacted = redact_sensitive_text(raw)
    assert "student@example.com" not in redacted
    assert "+213 555 123 456" not in redacted
    assert "123456789012" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_ID]" in redacted


@pytest.mark.high_level
def test_security__trace_sanitization_preserves_structure_while_redacting_sensitive_queries():
    trace = {"query_id": "q-1", "question": "email student@example.com id 12345678901", "timings_ms": {"total": 12.5}}
    safe = sanitize_trace(trace)
    assert safe["query_id"] == "q-1"
    assert safe["timings_ms"]["total"] == 12.5
    assert "student@example.com" not in safe["question"]
    assert "12345678901" not in safe["question"]
