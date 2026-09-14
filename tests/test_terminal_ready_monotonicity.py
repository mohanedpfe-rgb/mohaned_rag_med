from __future__ import annotations

from pathlib import Path

import pytest

from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.runtime_terminal_state_guard import install as install_terminal_guard


_FAILURE_TARGETS = [
    "FAILED",
    "FAILED_EXTRACTION",
    "FAILED_OCR",
    "FAILED_EMBEDDING",
    "FAILED_INDEXING",
    "DEGRADED_LEXICAL",
    "QUARANTINED",
    "RUNNING",
    "INDEXING",
    "VALIDATING_INDEX",
]


def _ready_store(tmp_path: Path) -> IngestionStateStore:
    install_terminal_guard()
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    store.upsert_document(
        {
            "document_id": "ready-doc",
            "content_hash": "ready-hash",
            "file_path": str(tmp_path / "ready.pdf"),
            "file_name": "ready.pdf",
            "file_size": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "modified_at": "2026-01-01T00:00:00+00:00",
            "ingestion_started_at": "2026-01-01T00:00:00+00:00",
            "current_stage": "READY",
            "current_page": 1,
            "total_pages": 1,
            "status": "READY",
            "parser_version": "pdf-extractor-v3",
            "ocr_config": "{}",
            "chunking_config": "{}",
            "embedding_model": "test",
            "embedding_dimension": 4,
            "index_state": "READY",
            "version_id": "ready-hash",
        }
    )
    return store


@pytest.mark.parametrize("target", _FAILURE_TARGETS)
def test_ready_document_cannot_transition_to_nonterminal_or_failure(tmp_path, target):
    store = _ready_store(tmp_path)
    with pytest.raises(RuntimeError, match="cannot"):
        store.transition_document_state(
            "ready-doc",
            target,
            current_page=1,
            total_pages=1,
            content_hash="ready-hash",
        )
    row = store.get_document("ready-doc")
    assert row["status"] == "READY"
    assert row["current_stage"] == "READY"
    assert row["index_state"] == "READY"


@pytest.mark.parametrize("target", _FAILURE_TARGETS)
def test_ready_document_cannot_be_downgraded_by_update_document(tmp_path, target):
    store = _ready_store(tmp_path)
    with pytest.raises(RuntimeError, match="cannot regress"):
        store.update_document(
            "ready-doc",
            status=target,
            current_stage=target,
            error="late injected failure",
        )
    row = store.get_document("ready-doc")
    assert row["status"] == "READY"
    assert row["current_stage"] == "READY"
    assert row["index_state"] == "READY"


def test_ready_document_can_be_explicitly_superseded(tmp_path):
    store = _ready_store(tmp_path)
    store.update_document(
        "ready-doc",
        status="SUPERSEDED",
        current_stage="SUPERSEDED",
        index_state="FAILED",
        error="replaced by a newer content version",
    )
    row = store.get_document("ready-doc")
    assert row["status"] == "SUPERSEDED"
    assert row["current_stage"] == "SUPERSEDED"
    assert row["index_state"] == "FAILED"


def test_completed_document_can_remain_terminal_but_not_fail(tmp_path):
    store = _ready_store(tmp_path)
    store.update_document("ready-doc", status="COMPLETED", current_stage="COMPLETED", index_state="READY")
    with pytest.raises(RuntimeError):
        store.update_document("ready-doc", status="FAILED", current_stage="FAILED")
    row = store.get_document("ready-doc")
    assert row["status"] == "READY"
    assert row["index_state"] == "READY"
