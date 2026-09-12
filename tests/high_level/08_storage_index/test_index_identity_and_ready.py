from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_storage__only_ready_document_version_is_searchable(clean_system, tmp_path):
    ready = write_minimal_pdf(tmp_path / "ready_marker.pdf", ["READY_ONLY_MARKER_42 is a controlled indexed term."])
    failed = tmp_path / "failed_marker.pdf"
    failed.write_bytes(b"not a valid PDF")

    ready_result = clean_system.ingest_file(ready)
    failed_result = clean_system.ingest_file(failed)
    assert_status(ready_result, {"READY"})
    assert str(failed_result.get("status") or "").upper() not in {"READY", "COMPLETED"}

    answer = clean_system.answer("What is READY_ONLY_MARKER_42?")
    assert str(answer.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}
    hits = answer.get("hits") or []
    assert hits
    hit_docs = {str(getattr(hit, "doc_id", "")) for hit in hits}
    assert str(ready_result.get("document_id") or ready_result.get("id")) in hit_docs


@pytest.mark.high_level
def test_storage__embedding_identity_matches_runtime_for_ready_index(clean_system):
    report = clean_system.vector_store.compatibility_report(clean_system.embedding_service.identity)
    assert isinstance(report, dict)
    assert report.get("compatible") is True or report.get("status") in {"compatible", "READY", "ok", "OK"}


@pytest.mark.high_level
def test_storage__ready_document_state_has_no_partial_index_flag(clean_system, ready_document):
    document_id = str((clean_system.state_store.get_by_path(str(ready_document)) or {}).get("document_id") or "")
    assert document_id
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert clean_system.state_store.is_ready_status(record.get("status"))
    assert str(record.get("index_state") or "").upper() == "READY"
    assert str(record.get("current_stage") or "").upper() in {"READY", "COMPLETED", "VALIDATING_INDEX"}
