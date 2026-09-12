from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_document_ready, assert_exact_path, assert_exact_status


@pytest.mark.high_level
def test_storage__only_ready_document_version_is_searchable(clean_system, tmp_path):
    ready = write_minimal_pdf(tmp_path / "ready_marker.pdf", ["READY_ONLY_MARKER_42 is a controlled indexed term."])
    failed = tmp_path / "failed_marker.pdf"
    failed.write_bytes(b"not a valid PDF")

    ready_result = clean_system.ingest_file(ready)
    failed_result = clean_system.ingest_file(failed)
    assert_exact_status(ready_result, "READY")
    assert_exact_status(failed_result, "FAILED")

    failed_document_id = str(failed_result.get("document_id") or failed_result.get("id") or "")
    assert failed_document_id
    record = clean_system.state_store.get_document(failed_document_id)
    assert record is not None
    assert str(record.get("status") or "").upper() == "FAILED"
    assert str(record.get("index_state") or "").upper() == "FAILED"
    with clean_system.state_store._connect() as connection:
        page_count = connection.execute(
            "SELECT COUNT(*) FROM pages WHERE document_id = ?",
            (failed_document_id,),
        ).fetchone()[0]
    assert int(page_count or 0) == 0

    answer = clean_system.answer("What is READY_ONLY_MARKER_42?")
    assert_exact_status(answer, "SUCCESS")
    assert_exact_path(answer, "PATH_A_EXTRACTIVE")
    hits = answer.get("hits") or []
    assert hits
    hit_docs = {str(getattr(hit, "doc_id", "")) for hit in hits}
    assert str(ready_result.get("document_id") or ready_result.get("id")) in hit_docs
    assert_citations_valid(answer)


@pytest.mark.high_level
def test_storage__embedding_identity_matches_runtime_for_ready_index(clean_system):
    report = clean_system.vector_store.compatibility_report(clean_system.embedding_service.identity)
    assert isinstance(report, dict)
    assert report.get("compatible") is True


@pytest.mark.high_level
def test_storage__ready_document_state_has_no_partial_index_flag(clean_system, ready_document):
    document = clean_system.state_store.get_by_path(str(ready_document)) or {}
    document_id = str(document.get("document_id") or "")
    assert_document_ready(clean_system, document_id)
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert str(record.get("current_stage") or "").upper() == "READY"
