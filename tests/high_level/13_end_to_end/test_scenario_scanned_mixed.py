from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_document_ready, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
@pytest.mark.slow
def test_e2e_scanned_mixed__ocr_document_becomes_exact_grounded_answer_and_native_pdf_remains_extractable(clean_system, scanned_document, tmp_path):
    scanned_result = clean_system.ingest_file(scanned_document)
    assert_exact_status(scanned_result, "READY")
    scanned_id = str(scanned_result.get("document_id") or scanned_result.get("id") or "")
    assert_document_ready(clean_system, scanned_id)

    with clean_system.state_store._connect() as connection:
        rows = connection.execute(
            "SELECT text, extraction_method, ocr_status FROM pages WHERE document_id = ? ORDER BY page_number",
            (scanned_id,),
        ).fetchall()
    assert len(rows) == 2
    assert all("OCR_CONTROLLED_MARKER" in str(row[0] or "") for row in rows)
    assert any(str(row[1] or "").upper() == "OCR" for row in rows)
    assert all(str(row[2] or "").strip() for row in rows)

    scanned_answer = clean_system.answer("What chronic disorder is contained in the scanned medical document?")
    assert_exact_status(scanned_answer, "SUCCESS")
    assert_exact_path(scanned_answer, "PATH_A_EXTRACTIVE")
    assert "diabetes" in str(scanned_answer.get("answer") or "").casefold()
    assert_citations_valid(scanned_answer)
    assert_grounded(scanned_answer)

    mixed = write_minimal_pdf(tmp_path / "native_text.pdf", ["NATIVE_MIXED_MARKER: HbA1c is used to assess glycemic control."])
    mixed_result = clean_system.ingest_file(mixed)
    assert str(mixed_result.get("status") or "").upper() == "READY"
    mixed_answer = clean_system.answer("What is NATIVE_MIXED_MARKER?")
    assert_exact_status(mixed_answer, "SUCCESS")
    assert_exact_path(mixed_answer, "PATH_A_EXTRACTIVE")
    assert "NATIVE_MIXED_MARKER" in str(mixed_answer.get("answer") or "")
    assert_citations_valid(mixed_answer)
    assert_grounded(mixed_answer)
