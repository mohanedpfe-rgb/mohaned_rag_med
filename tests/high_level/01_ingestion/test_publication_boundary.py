from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_document_ready, assert_exact_status


@pytest.mark.high_level
def test_ingestion__successful_publication_has_matching_ready_document_state_and_pages(clean_system, tmp_path):
    pdf = write_minimal_pdf(tmp_path / "publication_boundary.pdf", [
        "PAGE_ONE_PUBLICATION_MARKER: diabetes mellitus.",
        "PAGE_TWO_PUBLICATION_MARKER: HbA1c glycemic control.",
    ])
    result = clean_system.ingest_file(pdf)
    assert_exact_status(result, "READY")

    document_id = str(result.get("document_id") or "")
    record = assert_document_ready(clean_system, document_id)
    pages = clean_system.state_store.get_pages(document_id)
    assert [str(page.get("page_number")) for page in pages] == ["1", "2"]
    assert all(str(page.get("text") or "").strip() for page in pages)
    assert all(str(page.get("document_id") or "") == document_id for page in pages)
    assert str(record.get("index_state") or "").upper() == "READY"


@pytest.mark.high_level
def test_ingestion__failed_publication_returns_exact_failed_terminal_state_and_never_exposes_ready_index(clean_system, tmp_path, monkeypatch):
    pdf = write_minimal_pdf(tmp_path / "publication_failure.pdf", ["ATOMIC_FAIL_MARKER: controlled text."])

    def fail_validation(*args, **kwargs):
        return {"valid": False, "count": 0, "issues": ["forced publication validation failure"]}

    monkeypatch.setattr(clean_system.vector_store, "validate_document_index", fail_validation)
    result = clean_system.ingest_file(pdf)

    assert_exact_status(result, "FAILED")
    document_id = str(result.get("document_id") or "")
    assert document_id
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert str(record.get("index_state") or "").upper() == "FAILED_INDEXING"
    assert str(record.get("status") or "").upper() == "FAILED_INDEXING"
