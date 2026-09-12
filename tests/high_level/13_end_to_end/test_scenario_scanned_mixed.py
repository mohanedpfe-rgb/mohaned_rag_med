from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf


@pytest.mark.high_level
@pytest.mark.slow
def test_e2e_scanned_pdf__is_processed_or_safely_degraded_without_false_positive_text(clean_system, scanned_document, tmp_path):
    scanned_result = clean_system.ingest_file(scanned_document)
    status = str(scanned_result.get("status") or "").upper()
    assert status in {"READY", "COMPLETED", "DEGRADED_LEXICAL", "FAILED_OCR", "FAILED_EXTRACTION", "QUARANTINED"}

    if status in {"READY", "COMPLETED"}:
        document_id = str(scanned_result.get("document_id") or scanned_result.get("id") or "")
        record = clean_system.state_store.get_document(document_id)
        assert record is not None
        assert int(record.get("total_pages") or 0) == 2
        with clean_system.state_store._connect() as connection:
            rows = connection.execute("SELECT text, extraction_method FROM pages WHERE document_id = ? ORDER BY page_number", (document_id,)).fetchall()
        assert len(rows) == 2
        assert all(str(row[0] or "").strip() == "" or str(row[1] or "").upper() == "OCR" for row in rows)

    mixed = write_minimal_pdf(tmp_path / "mixed_followup.pdf", ["Diabetes mellitus is a chronic metabolic disorder."])
    mixed_result = clean_system.ingest_file(mixed)
    assert str(mixed_result.get("status") or "").upper() == "READY"
