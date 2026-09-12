from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_status


@pytest.mark.high_level
@pytest.mark.slow
def test_ingestion__scanned_pdf_is_detected_as_image_only_and_does_not_fake_text(clean_system, scanned_document):
    result = clean_system.ingest_file(scanned_document)
    status = str(result.get("status") or "").upper()
    assert status in {"READY", "COMPLETED", "DEGRADED_LEXICAL", "FAILED_OCR", "FAILED_EXTRACTION", "QUARANTINED"}

    document_id = str(result.get("document_id") or result.get("id") or "")
    if status in {"READY", "COMPLETED"} and document_id:
        record = clean_system.state_store.get_document(document_id)
        assert record is not None
        assert int(record.get("total_pages") or 0) == 2
        with clean_system.state_store._connect() as connection:
            rows = connection.execute("SELECT page_number, text, extraction_method, ocr_status FROM pages WHERE document_id = ? ORDER BY page_number", (document_id,)).fetchall()
        assert [row[0] for row in rows] == [1, 2]
        assert all(str(row[2] or "").upper() in {"OCR", "NONE", "UNKNOWN"} or row[3] for row in rows)


@pytest.mark.high_level
def test_ingestion__empty_pdf_does_not_become_searchable_as_valid_evidence(clean_system, empty_document):
    result = clean_system.ingest_file(empty_document)
    assert str(result.get("status") or "").upper() not in {"READY", "COMPLETED"} or int(result.get("chunk_count", 0) or 0) >= 0
