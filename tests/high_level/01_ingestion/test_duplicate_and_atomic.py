from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_duplicate_upload__second_copy_is_not_published_as_a_new_document(clean_system, temp_project_root):
    incoming = temp_project_root / "data" / "incoming"
    first_source = incoming / "duplicate_a.pdf"
    second_source = incoming / "duplicate_b.pdf"
    payload = write_minimal_pdf(first_source, ["Controlled duplicate marker: hyperglycemia."]).read_bytes()
    second_source.write_bytes(payload)

    first = clean_system.ingest_file(first_source)
    second = clean_system.ingest_file(second_source)

    assert_status(first, {"READY"})
    assert str(second.get("status") or "").upper() in {"SKIPPED", "DUPLICATE"}
    first_id = str(first.get("document_id") or first.get("id") or "")
    second_id = str(second.get("document_id") or second.get("id") or first_id)
    assert first_id
    assert second_id == first_id or second.get("status", "").upper() == "SKIPPED"

    digest = hashlib.sha256(payload).hexdigest()
    records = [row for row in clean_system.state_store.get_all_documents() if row.get("content_hash") == digest]
    assert len(records) == 1
    assert clean_system.state_store.is_ready_status(records[0]["status"])


@pytest.mark.high_level
def test_invalid_pdf__cannot_reach_ready_or_publish_ready_index(clean_system, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a PDF")

    result = clean_system.ingest_file(broken)

    assert isinstance(result, dict)
    assert str(result.get("status") or "").upper() not in {"READY", "COMPLETED"}
    document_id = str(result.get("document_id") or result.get("id") or "")
    if document_id:
        record = clean_system.state_store.get_document(document_id)
        assert record is not None
        assert not clean_system.state_store.is_ready_status(record.get("status"))
        assert str(record.get("index_state") or "").upper() != "READY"
