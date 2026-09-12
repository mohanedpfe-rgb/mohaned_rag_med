from __future__ import annotations

import pytest

from rag_project.api.med_evidence_api import (
    MAX_FEEDBACK_CHARS,
    MAX_METADATA_KEYS,
    MAX_QUERY_CHARS,
    _validate_feedback_payload,
    _validate_query_payload,
)


def test_query_payload_requires_non_empty_query():
    with pytest.raises(ValueError, match="query is required"):
        _validate_query_payload({"query": "   "})


def test_query_payload_enforces_query_and_metadata_limits():
    with pytest.raises(ValueError, match="character limit"):
        _validate_query_payload({"query": "x" * (MAX_QUERY_CHARS + 1)})

    with pytest.raises(ValueError, match="key limit"):
        _validate_query_payload(
            {
                "query": "test",
                "metadata_filter": {f"key-{i}": i for i in range(MAX_METADATA_KEYS + 1)},
            }
        )


def test_query_payload_rejects_non_object_metadata_filter():
    with pytest.raises(ValueError, match="metadata_filter must be a JSON object"):
        _validate_query_payload({"query": "test", "metadata_filter": ["ready"]})


def test_feedback_payload_bounds_untrusted_text():
    with pytest.raises(ValueError, match="feedback_text exceeds"):
        _validate_feedback_payload(
            {
                "query_id": "q-1",
                "feedback_type": "incorrect",
                "feedback_text": "x" * (MAX_FEEDBACK_CHARS + 1),
            }
        )


def test_feedback_payload_normalizes_valid_fields():
    assert _validate_feedback_payload(
        {
            "query_id": " q-1 ",
            "feedback_type": " incorrect ",
            "feedback_text": " useful correction ",
        }
    ) == ("q-1", "incorrect", " useful correction ")
