from __future__ import annotations

import pytest

from rag_project.ingestion import versioned_ingestor
from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import (
    assert_citations_valid,
    assert_document_ready,
    assert_exact_path,
    assert_exact_status,
    assert_grounded,
    assert_pipeline_authority,
)


@pytest.mark.high_level
def test_ingestion__changed_content_preserves_last_ready_version_when_reindex_fails(clean_system, tmp_path, monkeypatch):
    path = tmp_path / "versioned.pdf"
    write_minimal_pdf(path, ["VERSION_ONE_STABLE_MARKER: original clinical evidence."])
    first = clean_system.ingest_file(path)
    assert_exact_status(first, "READY")

    document_id = str(first.get("document_id") or first.get("id") or "")
    assert_document_ready(clean_system, document_id)
    original_pages = clean_system.state_store.get_pages(document_id)
    assert original_pages

    processed_path = clean_system.settings.processed_dir / path.name
    write_minimal_pdf(processed_path, ["VERSION_TWO_UNPUBLISHED_MARKER: replacement clinical evidence."])

    def fail_validation(document_id, version_id=None):
        return {"valid": False, "count": 0, "issues": ["forced validation failure"]}

    monkeypatch.setattr(clean_system.vector_store, "validate_document_index", fail_validation)
    replacement = clean_system.ingest_file(processed_path)

    assert_exact_status(replacement, "FAILED")
    preserved = clean_system.state_store.get_document(document_id)
    assert preserved is not None
    assert str(preserved.get("status") or "").upper() == "READY"
    assert str(preserved.get("index_state") or "").upper() == "READY"
    assert clean_system.state_store.get_pages(document_id) == original_pages


@pytest.mark.high_level
def test_ingestion__changed_content_publishes_new_version_and_supersedes_old(clean_system, tmp_path):
    path = tmp_path / "versioned_success.pdf"
    write_minimal_pdf(path, ["VERSION_ONE_OLD_MARKER: old controlled evidence."])
    first = clean_system.ingest_file(path)
    assert_exact_status(first, "READY")
    old_document_id = str(first.get("document_id") or first.get("id") or "")
    assert old_document_id

    processed_path = clean_system.settings.processed_dir / path.name
    write_minimal_pdf(processed_path, ["VERSION_TWO_NEW_MARKER: new controlled evidence."])

    replacement = clean_system.ingest_file(processed_path)
    assert_exact_status(replacement, "READY")
    new_document_id = str(replacement.get("document_id") or replacement.get("id") or "")
    assert new_document_id
    assert new_document_id != old_document_id
    assert replacement.get("versioned_replacement") is True
    assert replacement.get("previous_version_retired") is True

    new_record = assert_document_ready(clean_system, new_document_id)
    assert str(new_record.get("file_name") or "") == path.name

    old_record = clean_system.state_store.get_document(old_document_id)
    assert old_record is not None
    assert str(old_record.get("status") or "").upper() == "SUPERSEDED"
    assert str(old_record.get("index_state") or "").upper() == "FAILED"
    assert clean_system.state_store.get_pages(old_document_id) == []

    result = clean_system.answer("What does VERSION_TWO_NEW_MARKER state?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert_citations_valid(result)
    assert_grounded(result)
    assert_pipeline_authority(result)
    hit_document_ids = {str(getattr(hit, "doc_id", "")) for hit in result.get("hits") or []}
    assert new_document_id in hit_document_ids
    assert old_document_id not in hit_document_ids


@pytest.mark.high_level
def test_ingestion__replacement_retirement_failure_rolls_back_to_last_known_good(monkeypatch, clean_system, tmp_path):
    path = tmp_path / "versioned_retirement_failure.pdf"
    write_minimal_pdf(path, ["VERSION_ONE_ROLLBACK_MARKER: last known good evidence."])
    first = clean_system.ingest_file(path)
    assert_exact_status(first, "READY")
    old_document_id = str(first.get("document_id") or first.get("id") or "")
    assert old_document_id

    processed_path = clean_system.settings.processed_dir / path.name
    write_minimal_pdf(processed_path, ["VERSION_TWO_ABORTED_MARKER: should never become published evidence."])

    def fail_retirement(*args, **kwargs):
        raise RuntimeError("forced retirement failure")

    monkeypatch.setattr(versioned_ingestor, "_retire_previous_version", fail_retirement)

    result = clean_system.ingest_file(processed_path)
    assert_exact_status(result, "FAILED")
    assert result.get("versioned_replacement") is True
    assert result.get("previous_document_id") == old_document_id

    old_record = clean_system.state_store.get_document(old_document_id)
    assert old_record is not None
    assert str(old_record.get("status") or "").upper() == "READY"
    assert str(old_record.get("index_state") or "").upper() == "READY"

    replacement_document_id = str(result.get("document_id") or "")
    assert replacement_document_id
    replacement_record = clean_system.state_store.get_document(replacement_document_id)
    assert replacement_record is not None
    assert str(replacement_record.get("status") or "").upper() == "FAILED_INDEXING"
    assert str(replacement_record.get("index_state") or "").upper() == "FAILED"
    assert clean_system.state_store.get_pages(replacement_document_id) == []

    answer = clean_system.answer("What does VERSION_ONE_ROLLBACK_MARKER state?")
    assert_exact_status(answer, "SUCCESS")
    assert_exact_path(answer, "PATH_A_EXTRACTIVE")
    assert_citations_valid(answer)
    assert_grounded(answer)
    assert_pipeline_authority(answer)
    hit_document_ids = {str(getattr(hit, "doc_id", "")) for hit in answer.get("hits") or []}
    assert old_document_id in hit_document_ids
    assert replacement_document_id not in hit_document_ids
