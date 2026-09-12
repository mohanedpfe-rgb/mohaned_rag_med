from __future__ import annotations

import pytest


@pytest.mark.high_level
@pytest.mark.slow
def test_ingestion__image_heavy_scanned_pdf_activates_ocr_metadata_and_never_fabricates_text(clean_system, scanned_document):
    result = clean_system.ingest_file(scanned_document)
    assert str(result.get("status") or "").upper() == "FAILED"

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
        assert str(method or "").upper() in {"OCR", "NONE", "UNKNOWN"}
        if str(method or "").upper() == "OCR":
            assert str(ocr_status or "").strip()
        assert "clinical" not in str(text or "").casefold()
        assert "diabetes" not in str(text or "").casefold()

    answer = clean_system.answer("What clinical fact is contained in this image-only document?")
    assert str(answer.get("status") or "").upper() == "NOT_SUPPORTED"
    assert answer.get("citations") == []


@pytest.mark.high_level
def test_ingestion__empty_pdf_cannot_reach_ready_or_produce_a_positive_answer(clean_system, empty_document):
    result = clean_system.ingest_file(empty_document)
    assert str(result.get("status") or "").upper() == "FAILED"

    document_id = str(result.get("document_id") or result.get("id") or "")
    assert document_id
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert str(record.get("status") or "").upper() == "FAILED"
    assert str(record.get("index_state") or "").upper() == "FAILED"

    answer = clean_system.answer("What unique clinical fact is contained in the empty PDF?")
    assert str(answer.get("status") or "").upper() == "NOT_SUPPORTED"
    assert answer.get("citations") == []
    assert not (answer.get("claims") or [])
