from __future__ import annotations

import hashlib

import pytest

from tests.high_level.conftest import write_minimal_pdf


@pytest.mark.high_level
def test_ingestion__duplicate_is_archived_and_not_reindexed(clean_system, temp_project_root):
    incoming = temp_project_root / "data" / "incoming"
    first_source = incoming / "duplicate_a.pdf"
    second_source = incoming / "duplicate_b.pdf"
    payload = write_minimal_pdf(first_source, ["Controlled duplicate marker: hyperglycemia."]).read_bytes()
    second_source.write_bytes(payload)

    first = clean_system.ingest_file(first_source)
    second = clean_system.ingest_file(second_source)

    assert str(first.get("status") or "").upper() == "READY"
    assert str(second.get("status") or "").upper() == "SKIPPED"
    first_id = str(first.get("document_id") or first.get("id") or "")
    second_id = str(second.get("document_id") or second.get("id") or "")
    assert first_id and second_id == first_id

    archived = second.get("archive_path")
    assert archived
    assert str(archived).startswith(str(temp_project_root / "data" / "archive"))
    assert __import__("pathlib").Path(archived).is_file()

    digest = hashlib.sha256(payload).hexdigest()
    records = [row for row in clean_system.state_store.get_all_documents() if row.get("content_hash") == digest]
    assert len(records) == 1
    assert str(records[0].get("status") or "").upper() == "READY"
    assert str(records[0].get("index_state") or "").upper() == "READY"


@pytest.mark.high_level
def test_ingestion__invalid_pdf_is_failed_and_never_published(clean_system, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a PDF")

    result = clean_system.ingest_file(broken)

    assert str(result.get("status") or "").upper() == "FAILED"
    document_id = str(result.get("document_id") or result.get("id") or "")
    assert document_id
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert str(record.get("status") or "").upper() == "FAILED"
    assert str(record.get("index_state") or "").upper() == "FAILED"
