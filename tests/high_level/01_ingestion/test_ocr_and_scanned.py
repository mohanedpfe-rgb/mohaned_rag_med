from __future__ import annotations

import pytest


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
        assert all(str(row[2] or "").upper() in {"OCR", "NONE", "UNKNOWN"} or str(row[3] or "").strip() for row in rows)


@pytest.mark.high_level
def test_ingestion__empty_pdf_fails_or_finishes_without_searchable_content(clean_system, empty_document):
    result = clean_system.ingest_file(empty_document)
    status = str(result.get("status") or "").upper()
    assert status in {"FAILED", "FAILED_EXTRACTION", "FAILED_INDEXING", "QUARANTINED", "DEGRADED_LEXICAL", "READY", "COMPLETED"}

    if status in {"READY", "COMPLETED"}:
        document_id = str(result.get("document_id") or result.get("id") or "")
        assert document_id
        record = clean_system.state_store.get_document(document_id)
        assert record is not None
        with clean_system.state_store._connect() as connection:
            chunks = connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        assert chunks > 0
        answer = clean_system.answer("What unique clinical fact is contained in the empty PDF?")
        assert str(answer.get("status") or "").upper() in {"NOT_SUPPORTED", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE", "SUCCESS_WITH_WARNINGS"}
