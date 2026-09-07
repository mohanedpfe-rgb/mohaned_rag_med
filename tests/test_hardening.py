from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import fitz
import pytest

from rag_project.app.rag_system import RAGSystem
from rag_project.citations.citation_manager import CitationManager
from rag_project.configuration.settings import Settings
from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.ingestion.file_monitor import FileMonitor
from rag_project.parsing.pdf_extractor import PDFExtractor


def test_settings_from_env_loads_project_dotenv(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("EMBEDDING_MODEL=env-model\nEMBEDDING_TEST_MODE=true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_TEST_MODE", raising=False)
    os.environ["PROJECT_ROOT"] = str(project_root)
    loaded = Settings.from_env()
    assert loaded.embedding_model == "env-model"
    assert loaded.embedding_test_mode is True


def test_state_store_survives_restart(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    first = IngestionStateStore(database)
    first.upsert_document(
        {
            "document_id": "doc-1",
            "content_hash": "hash-1",
            "file_path": str(tmp_path / "a.pdf"),
            "file_name": "a.pdf",
            "file_size": 10,
            "embedding_model": "test",
        }
    )
    first.upsert_page("doc-1", 1, extraction_status="COMPLETED", text="page text")

    second = IngestionStateStore(database)
    assert second.get_by_hash("hash-1")["document_id"] == "doc-1"
    assert second.get_pages("doc-1")[0]["text"] == "page text"


def test_state_store_claims_leases_and_recovers_expired_workers(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    store = IngestionStateStore(database)
    store.upsert_document(
        {
            "document_id": "doc-lease",
            "content_hash": "hash-lease",
            "file_path": str(tmp_path / "lease.pdf"),
            "file_name": "lease.pdf",
            "file_size": 10,
            "embedding_model": "test",
        }
    )

    assert store.claim_document("doc-lease", "worker-a", lease_seconds=900)
    assert not store.claim_document("doc-lease", "worker-b", lease_seconds=900)
    assert store.heartbeat_document("doc-lease", "worker-a", lease_seconds=900)
    store.update_document(
        "doc-lease",
        lease_expires_at="2000-01-01T00:00:00+00:00",
    )
    assert store.recover_stale_documents() == 1
    assert store.get_document("doc-lease")["status"] == "INTERRUPTED"
    assert store.claim_document("doc-lease", "worker-b", lease_seconds=900)


def test_state_store_recovers_after_child_process_termination(tmp_path: Path):
    database = tmp_path / "child-state.sqlite3"
    child_code = (
        "import sys,time; "
        "from rag_project.ingestion.state_store import IngestionStateStore; "
        "store=IngestionStateStore(sys.argv[1]); "
        "store.upsert_document({'document_id':'child-doc','content_hash':'child-hash',"
        "'file_path':'child.pdf','file_name':'child.pdf','file_size':1,"
        "'embedding_model':'test'}); "
        "assert store.claim_document('child-doc','child-worker',lease_seconds=900); "
        "time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code, str(database)])
    try:
        time.sleep(1)
        store = IngestionStateStore(database)
        assert store.get_document("child-doc")["lease_owner"] == "child-worker"
        child.terminate()
        child.wait(timeout=10)
        store.update_document(
            "child-doc", lease_expires_at="2000-01-01T00:00:00+00:00"
        )
        assert store.recover_stale_documents() == 1
        assert store.claim_document("child-doc", "replacement-worker", lease_seconds=900)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_separate_process_interruption_keeps_building_version_hidden(tmp_path: Path):
    vector_dir = tmp_path / "vectors"
    state_db = tmp_path / "state.sqlite3"
    setup_code = (
        "import sys; "
        "from rag_project.storage.vector_store import VectorStore; "
        "store=VectorStore(sys.argv[1]); "
        "store.add_documents(['old evidence'], "
        "[{'document_id':'doc','chunk_id':'old-0','page_numbers':[1],"
        "'version_id':'old','index_state':'READY'}], "
        "[[0.1]*32], ['old-0'])"
    )
    subprocess.run(
        [sys.executable, "-c", setup_code, str(vector_dir)],
        check=True,
        timeout=30,
    )
    child_code = (
        "import sys,time; "
        "from rag_project.storage.vector_store import VectorStore; "
        "store=VectorStore(sys.argv[1]); "
        "store.add_documents(['new evidence'], "
        "[{'document_id':'doc','chunk_id':'new-0','page_numbers':[2],"
        "'version_id':'new','index_state':'BUILDING'}], "
        "[[0.2]*32], ['new-0']); "
        "time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code, str(vector_dir)])
    try:
        time.sleep(2)
        child.terminate()
        child.wait(timeout=10)
        from rag_project.storage.vector_store import VectorStore

        store = VectorStore(vector_dir)
        old = store.search_lexical("old evidence")
        new = store.search_lexical("new evidence")
        assert old["ids"][0] == ["old-0"]
        assert "new-0" not in new["ids"][0]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_embedding_failures_are_not_silently_replaced(monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr("rag_project.embeddings.embedding_service.requests.post", fail)
    service = EmbeddingService("http://offline", "model", retries=1)
    with pytest.raises(RuntimeError, match="Embedding service failed"):
        service.embed_texts(["text"])


def test_test_embeddings_require_explicit_mode():
    service = EmbeddingService("http://unused", "test", test_mode=True)
    assert len(service.embed_texts(["text"])[0]) == 32


def test_file_monitor_marks_content_duplicates_not_duplicate_names(tmp_path: Path):
    first = tmp_path / "one.pdf"
    second = tmp_path / "renamed.pdf"
    first.write_bytes(b"same content")
    second.write_bytes(b"same content")
    results = FileMonitor(tmp_path).scan()
    assert sum(bool(item.get("duplicate")) for item in results) == 2
    assert FileMonitor(tmp_path).scan()[0]["status"] == "unchanged"


def test_settings_from_env_accepts_common_truthy_values(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("EMBEDDING_TEST_MODE", "1")
    monkeypatch.setenv("NEIGHBOR_EXPANSION", "yes")
    loaded = Settings.from_env()
    assert loaded.embedding_test_mode is True
    assert loaded.neighbor_expansion is True


def test_settings_load_generation_latency_budget(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("GENERATION_LATENCY_BUDGET_SECONDS", "45")
    loaded = Settings.from_env()
    assert loaded.generation_latency_budget_seconds == 45.0


def test_file_monitor_creates_missing_directory(tmp_path: Path):
    base_dir = tmp_path / "missing" / "nested"
    results = FileMonitor(base_dir).scan()
    assert results == []
    assert (base_dir / ".file_index.json").exists()


def test_ingestion_rejects_non_pdf_and_missing_files(tmp_path: Path):
    settings = Settings(
        project_root=tmp_path,
        incoming_dir=tmp_path / "incoming",
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        archive_dir=tmp_path / "archive",
        vector_db_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        ingestion_db_path=tmp_path / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
    )
    system = RAGSystem(settings)
    with pytest.raises(ValueError, match="Unsupported or missing PDF"):
        system.ingest_file(tmp_path / "not-a-pdf.txt")


def test_empty_pdf_is_failed_and_quarantined(tmp_path: Path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf_path = incoming / "empty.pdf"
    document = fitz.open()
    document.new_page()
    document.save(str(pdf_path))
    document.close()
    settings = Settings(
        project_root=tmp_path,
        incoming_dir=incoming,
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        archive_dir=tmp_path / "archive",
        vector_db_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        ingestion_db_path=tmp_path / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
    )
    result = RAGSystem(settings).ingest_file(pdf_path)
    assert result["status"] == "failed"
    assert (settings.failed_dir / pdf_path.name).exists()
    assert "No extractable text" in result["error"]


def test_quarantine_copy_failure_is_reported_without_masking_ingestion_error(
    tmp_path: Path, monkeypatch
):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf_path = incoming / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nnot a valid cross-reference table")
    settings = Settings(
        project_root=tmp_path,
        incoming_dir=incoming,
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        archive_dir=tmp_path / "archive",
        vector_db_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        ingestion_db_path=tmp_path / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
    )
    monkeypatch.setattr(
        "rag_project.app.rag_system.shutil.copy2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("simulated quarantine write failure")
        ),
    )
    result = RAGSystem(settings).ingest_file(pdf_path)
    assert result["status"] == "failed"
    assert "simulated quarantine write failure" in result["quarantine_error"]
    assert "cannot" in result["error"].lower() or "file" in result["error"].lower()


def test_malformed_pdf_raises_explicit_parser_error(tmp_path: Path):
    pdf_path = tmp_path / "malformed.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nnot a valid cross-reference table")
    with pytest.raises(Exception):
        PDFExtractor().extract(pdf_path)


def test_ingestion_is_content_idempotent_and_checkpointed(tmp_path: Path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf_path = incoming / "original.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A durable ingestion checkpoint test document.")
    document.save(str(pdf_path))
    document.close()

    settings = Settings(
        project_root=tmp_path,
        incoming_dir=incoming,
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        archive_dir=tmp_path / "archive",
        vector_db_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        ingestion_db_path=tmp_path / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
    )
    system = RAGSystem(settings)
    first = system.ingest_file(pdf_path)
    assert first["status"] == "success"
    metrics = system.state_store.get_document(first["document_id"])["ingestion_metrics"]
    assert "extraction" in metrics
    assert "indexing" in metrics

    duplicate = incoming / "renamed-copy.pdf"
    (settings.processed_dir / "original.pdf").replace(duplicate)
    second = system.ingest_file(duplicate)
    assert second["status"] == "skipped"
    pages = system.state_store.get_pages(first["document_id"])
    assert pages[0]["extraction_status"] == "COMPLETED"


def test_modified_document_replaces_old_indexed_content(tmp_path: Path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf_path = incoming / "book.pdf"

    def write_pdf(path: Path, text: str) -> None:
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), text)
        document.save(str(path))
        document.close()

    settings = Settings(
        project_root=tmp_path,
        incoming_dir=incoming,
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        archive_dir=tmp_path / "archive",
        vector_db_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        ingestion_db_path=tmp_path / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
    )
    system = RAGSystem(settings)
    write_pdf(pdf_path, "OLD_CONTENT_UNIQUE " + "stable text " * 30)
    first = system.ingest_file(pdf_path)

    processed_path = settings.processed_dir / "book.pdf"
    write_pdf(processed_path, "NEW_CONTENT_UNIQUE " + "replacement text " * 30)
    second = system.ingest_file(processed_path)

    records = system.vector_store.get_documents()
    texts = records.get("documents", [])
    assert first["document_id"] == second["document_id"]
    assert second["status"] == "success"
    assert any("NEW_CONTENT_UNIQUE" in text for text in texts)
    assert not any("OLD_CONTENT_UNIQUE" in text for text in texts)


def test_failed_replacement_preserves_previous_ready_version(tmp_path: Path, monkeypatch):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf_path = incoming / "book.pdf"

    def write_pdf(path: Path, text: str) -> None:
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), text)
        document.save(str(path))
        document.close()

    settings = Settings(
        project_root=tmp_path,
        incoming_dir=incoming,
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        archive_dir=tmp_path / "archive",
        vector_db_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        ingestion_db_path=tmp_path / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
    )
    system = RAGSystem(settings)
    write_pdf(pdf_path, "OLD_CONTENT_UNIQUE " + "stable text " * 30)
    first = system.ingest_file(pdf_path)
    assert first["status"] == "success"

    processed_path = settings.processed_dir / "book.pdf"
    write_pdf(processed_path, "NEW_CONTENT_UNIQUE " + "replacement text " * 30)
    def fail_index(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(system.vector_store, "add_documents", fail_index)
    monkeypatch.setattr(system.vector_store, "add_lexical_documents", fail_index)
    failed = system.ingest_file(processed_path)

    assert failed["status"] == "failed"
    lexical = system.vector_store.search_lexical("OLD_CONTENT_UNIQUE")
    assert lexical["ids"][0]
    assert first["document_id"] in lexical["metadatas"][0][0]["document_id"]


def test_activation_failure_keeps_previous_version_searchable(tmp_path: Path, monkeypatch):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf_path = incoming / "book.pdf"

    def write_pdf(path: Path, text: str) -> None:
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), text)
        document.save(str(path))
        document.close()

    settings = Settings(
        project_root=tmp_path,
        incoming_dir=incoming,
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        archive_dir=tmp_path / "archive",
        vector_db_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        ingestion_db_path=tmp_path / "ingestion.sqlite3",
        embedding_test_mode=True,
        embedding_model="test",
    )
    system = RAGSystem(settings)
    write_pdf(pdf_path, "OLD_ACTIVATION_CONTENT " + "stable text " * 30)
    first = system.ingest_file(pdf_path)
    assert first["status"] == "success"

    processed_path = settings.processed_dir / "book.pdf"
    write_pdf(processed_path, "NEW_ACTIVATION_CONTENT " + "replacement text " * 30)
    original_activate = system.vector_store.set_version_index_state

    def fail_new_activation(document_id, version_id, state):
        if state == "READY" and version_id != first["document_id"]:
            raise RuntimeError("activation interrupted")
        return original_activate(document_id, version_id, state)

    monkeypatch.setattr(system.vector_store, "set_version_index_state", fail_new_activation)
    result = system.ingest_file(processed_path)

    assert result["status"] == "failed"
    old_results = system.vector_store.search_lexical("OLD_ACTIVATION_CONTENT")
    assert old_results["ids"][0]
    new_results = system.vector_store.search_lexical("NEW_ACTIVATION_CONTENT")
    assert new_results["ids"][0] == []


def test_citation_validation_rejects_unknown_chunk_id():
    class Hit:
        doc_id = "doc-a"
        text = "evidence"
        metadata = {"document_id": "doc-a", "chunk_id": "chunk-1", "page_numbers": [1]}

    valid = [{"document_id": "doc-a", "chunk_id": "chunk-1", "page_numbers": [1]}]
    fabricated = [{"document_id": "doc-a", "chunk_id": "chunk-999", "page_numbers": [1]}]
    assert len(CitationManager.validate(valid, [Hit()])) == 1
    assert CitationManager.validate(fabricated, [Hit()]) == []
