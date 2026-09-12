from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_document_ready, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
@pytest.mark.slow
def test_ingestion__image_heavy_scanned_pdf_activates_ocr_and_becomes_grounded_searchable_content(clean_system, scanned_document):
    result = clean_system.ingest_file(scanned_document)
    assert str(result.get("status") or "").upper() == "READY"

    document_id = str(result.get("document_id") or result.get("id") or "")
    assert_document_ready(clean_system, document_id)

    with clean_system.state_store._connect() as connection:
        rows = connection.execute(
            "SELECT page_number, text, extraction_method, ocr_status FROM pages WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()

    assert [row[0] for row in rows] == [1, 2]
    assert all("OCR_CONTROLLED_MARKER" in str(row[1] or "") for row in rows)
    assert any(str(row[2] or "").upper() == "OCR" for row in rows)
    assert all(str(row[3] or "").strip() for row in rows)

    answer = clean_system.answer("What chronic disorder is contained in the scanned medical document?")
    assert_exact_status(answer, "SUCCESS")
    assert_exact_path(answer, "PATH_A_EXTRACTIVE")
    assert "diabetes" in str(answer.get("answer") or "").casefold()
    assert_citations_valid(answer)
    assert_grounded(answer)


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
