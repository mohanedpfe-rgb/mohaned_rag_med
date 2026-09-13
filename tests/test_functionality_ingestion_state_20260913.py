from pathlib import Path

from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.runtime_functionality_state_fix import _wrap_document_ready


def _store(tmp_path: Path) -> IngestionStateStore:
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    store.upsert_document({
        "document_id": "doc-1",
        "content_hash": "hash-1",
        "file_path": "/tmp/book.pdf",
        "file_name": "book.pdf",
        "file_size": 100,
        "current_stage": "READY",
        "current_page": 2,
        "total_pages": 2,
        "status": "READY",
        "index_state": "READY",
    })
    return store


def test_interrupted_transition_removes_ready_index_state(tmp_path):
    store = _store(tmp_path)

    store.transition_document_state("doc-1", "INTERRUPTED")

    record = store.get_document("doc-1")
    assert record["status"] == "INTERRUPTED"
    assert record["current_stage"] == "INTERRUPTED"
    assert record["index_state"] == "FAILED"


def test_recovering_transition_removes_ready_index_state(tmp_path):
    store = _store(tmp_path)

    store.transition_document_state("doc-1", "RECOVERING")

    record = store.get_document("doc-1")
    assert record["status"] == "RECOVERING"
    assert record["current_stage"] == "RECOVERING"
    assert record["index_state"] == "FAILED"


def test_document_ready_requires_both_status_and_ready_index_state(tmp_path):
    store = _store(tmp_path)
    store.update_document("doc-1", status="READY", index_state="FAILED")

    original = lambda self, document_id: True
    wrapped = _wrap_document_ready(original)
    assert wrapped(store, "doc-1") is False


def test_document_ready_is_true_for_ready_status_and_ready_index(tmp_path):
    store = _store(tmp_path)

    wrapped = _wrap_document_ready(lambda self, document_id: False)
    assert wrapped(store, "doc-1") is True
