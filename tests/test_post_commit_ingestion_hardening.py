from __future__ import annotations

import fitz

from rag_project.configuration.settings import Settings
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.runtime_ready_publication_fix import install as install_ready_guards
from rag_project.storage.vector_store import VectorStore


def _settings(root):
    return Settings(
        project_root=root,
        incoming_dir=root / "incoming",
        processed_dir=root / "processed",
        failed_dir=root / "failed",
        archive_dir=root / "archive",
        vector_db_dir=root / "vectors",
        log_dir=root / "logs",
        ingestion_db_path=root / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
        ocr_enabled=False,
    )


def test_ready_transition_survives_audit_event_failure(tmp_path, monkeypatch):
    install_ready_guards()
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    document_id = "doc-ready-event-failure"
    store.upsert_document(
        {
            "document_id": document_id,
            "content_hash": "hash-ready",
            "file_path": str(tmp_path / "medical.pdf"),
            "file_name": "medical.pdf",
            "file_size": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "modified_at": "2026-01-01T00:00:00+00:00",
            "ingestion_started_at": "2026-01-01T00:00:00+00:00",
            "current_stage": "VALIDATING_INDEX",
            "current_page": 112,
            "total_pages": 112,
            "status": "RUNNING",
            "parser_version": "pdf-extractor-v3",
            "ocr_config": "{}",
            "chunking_config": "{}",
            "embedding_model": "test",
            "index_state": "PENDING",
            "version_id": "hash-ready",
        }
    )

    def fail_event(*args, **kwargs):
        raise RuntimeError("injected audit-event failure")

    monkeypatch.setattr(store, "record_event", fail_event)
    store.transition_document_state(
        document_id,
        "READY",
        current_page=112,
        total_pages=112,
        content_hash="hash-ready",
        index_state="READY",
    )

    row = store.get_document(document_id)
    assert row is not None
    assert row["status"] == "READY"
    assert row["current_stage"] == "READY"
    assert row["index_state"] == "READY"


def test_release_failure_cannot_escape_cleanup_boundary(tmp_path, monkeypatch):
    install_ready_guards()
    store = IngestionStateStore(tmp_path / "state.sqlite3")

    def fail_connect():
        raise RuntimeError("injected release failure")

    monkeypatch.setattr(store, "_connect", fail_connect)
    assert store.release_document("missing", "worker") is False


def test_112_page_document_reaches_ready_in_test_embedding_mode(tmp_path):
    settings = _settings(tmp_path)
    settings.incoming_dir.mkdir(parents=True)
    pdf_path = settings.incoming_dir / "DC_endocrino_112_page_regression.pdf"

    document = fitz.open()
    try:
        for page_number in range(1, 113):
            page = document.new_page()
            page.insert_text(
                (72, 72),
                f"Endocrinology regression page {page_number}. "
                "This page contains stable searchable medical source text for long-document ingestion testing.",
            )
        document.save(str(pdf_path))
    finally:
        document.close()

    from rag_project.app.rag_system import RAGSystem

    system = RAGSystem(settings)
    result = system.ingest_file(pdf_path)
    assert str(result.get("status", "")).upper() in {"SUCCESS", "READY", "COMPLETED"}, result

    document_id = str(result["document_id"])
    row = system.state_store.get_document(document_id)
    assert row is not None
    assert row["status"] == "READY"
    assert row["index_state"] == "READY"
    assert int(row["current_page"]) == 112
    assert int(row["total_pages"]) == 112
    assert int(result["embedding_count"]) > 0
    validation = system.vector_store.validate_document_index(document_id, str(row["version_id"]))
    assert validation["valid"] is True, validation
    assert validation["count"] == result["embedding_count"]
