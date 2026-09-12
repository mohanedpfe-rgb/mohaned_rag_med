from __future__ import annotations

import pytest


@pytest.mark.high_level
@pytest.mark.slow
def test_ingestion__scanned_pdf_never_fabricates_searchable_text_without_real_ocr(clean_system, scanned_document):
    result = clean_system.ingest_file(scanned_document)
    status = str(result.get("status") or "").upper()
    assert status in {"READY", "COMPLETED", "DEGRADED_LEXICAL", "FAILED_OCR", "FAILED_EXTRACTION", "QUARANTINED"}

    document_id = str(result.get("document_id") or result.get("id") or "")
    assert document_id
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert int(record.get("total_pages") or 0) == 2

    with clean_system.state_store._connect() as connection:
        rows = connection.execute(
            "SELECT page_number, text, extraction_method, ocr_status FROM pages WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()

    assert [row[0] for row in rows] == [1, 2]
    for _, text, method, ocr_status in rows:
        assert str(method or "").upper() in {"OCR", "NONE", "UNKNOWN"} or str(ocr_status or "").strip()
        if str(method or "").upper() == "OCR":
            assert str(ocr_status or "").strip(), "OCR-derived pages must expose an explicit OCR state"
        # The blank 1x1 image fixture contains no encoded clinical text.
        assert "clinical" not in str(text or "").casefold()
        assert "diabetes" not in str(text or "").casefold()

    answer = clean_system.answer("What clinical fact is contained in this image-only document?")
    answer_status = str(answer.get("status") or "").upper()
    assert answer_status in {"NOT_SUPPORTED", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE"}
    assert answer.get("citations") == []


@pytest.mark.high_level
def test_ingestion__empty_pdf_cannot_become_a_positive_search_answer(clean_system, empty_document):
    result = clean_system.ingest_file(empty_document)
    status = str(result.get("status") or "").upper()
    assert status in {"FAILED", "FAILED_EXTRACTION", "FAILED_INDEXING", "QUARANTINED", "DEGRADED_LEXICAL", "READY", "COMPLETED"}

    document_id = str(result.get("document_id") or result.get("id") or "")
    if not document_id:
        return

    record = clean_system.state_store.get_document(document_id)
    assert record is not None

    answer = clean_system.answer("What unique clinical fact is contained in the empty PDF?")
    answer_status = str(answer.get("status") or "").upper()
    assert answer_status in {"NOT_SUPPORTED", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE"}
    assert answer.get("citations") == []
    assert not (answer.get("claims") or [])
