from __future__ import annotations

from pathlib import Path

from rag_project.app.rag_system import RAGSystem
from rag_project.configuration.settings import Settings
from rag_project.ingestion import robust_ingestor
import fitz


def _settings(root: Path) -> Settings:
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
        ingestion_lease_seconds=60,
    )


def _pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        doc.save(str(path))
    finally:
        doc.close()


def test_failure_after_processed_move_quarantines_new_file_then_restores_old(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    old_target = settings.processed_dir / "rollback-after-move.pdf"
    _pdf(old_target, "OLD GOOD PROCESSED VERSION")
    old_bytes = old_target.read_bytes()

    incoming = settings.incoming_dir / old_target.name
    _pdf(incoming, "NEW FAILED VERSION")
    new_bytes = incoming.read_bytes()

    system = RAGSystem(settings)
    original_set_version = system.vector_store.set_version_index_state
    calls = {"ready": 0}

    def fail_new_ready(document_id, version_id, state):
        if state == "READY" and calls["ready"] == 0:
            calls["ready"] += 1
            raise RuntimeError("injected READY publication failure after file move")
        return original_set_version(document_id, version_id, state)

    monkeypatch.setattr(system.vector_store, "set_version_index_state", fail_new_ready)
    result = robust_ingestor.robust_ingest_file(system, incoming)

    assert str(result["status"]).lower() == "failed"
    assert old_target.exists()
    assert old_target.read_bytes() == old_bytes, "old good processed file was damaged by rollback"
    failed_files = list(settings.failed_dir.glob("*.pdf"))
    assert failed_files, "new failed input must be quarantined"
    assert any(path.read_bytes() == new_bytes for path in failed_files)
