from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from rag_project.ingestion.atomic_claim import ensure_and_claim
from rag_project.ingestion.state_store import IngestionStateStore


def _values():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "document_id": "doc-1",
        "content_hash": "hash-1",
        "file_path": "/tmp/a.pdf",
        "file_name": "a.pdf",
        "file_size": 10,
        "created_at": now,
        "modified_at": now,
        "ingestion_started_at": now,
        "current_stage": "DISCOVERED",
        "current_page": 0,
        "total_pages": 0,
        "status": "RUNNING",
        "parser_version": "test",
        "ocr_config": "{}",
        "chunking_config": "{}",
        "embedding_model": "test",
        "index_state": "PENDING",
        "version_id": "hash-1",
    }


def test_first_worker_atomically_claims(tmp_path):
    store = IngestionStateStore(tmp_path / "state.db")
    assert ensure_and_claim(store, _values(), "worker-1", 300) is True
    row = store.get_document("doc-1")
    assert row["lease_owner"] == "worker-1"
    assert row["status"] == "RUNNING"


def test_second_live_worker_cannot_overwrite_claim(tmp_path):
    store = IngestionStateStore(tmp_path / "state.db")
    assert ensure_and_claim(store, _values(), "worker-1", 300) is True
    assert ensure_and_claim(store, _values(), "worker-2", 300) is False
    assert store.get_document("doc-1")["lease_owner"] == "worker-1"


def test_expired_worker_can_reclaim(tmp_path):
    store = IngestionStateStore(tmp_path / "state.db")
    values = _values()
    assert ensure_and_claim(store, values, "worker-1", 1) is True
    expired = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    with store._connect() as connection:
        connection.execute(
            "UPDATE documents SET lease_expires_at = ? WHERE document_id = ?",
            (expired, "doc-1"),
        )
    assert ensure_and_claim(store, values, "worker-2", 300) is True
    assert store.get_document("doc-1")["lease_owner"] == "worker-2"
