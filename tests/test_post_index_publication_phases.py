from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3

import fitz
import pytest

from rag_project.configuration.settings import Settings
from rag_project.ingestion.atomic_claim import ensure_and_claim
from rag_project.ingestion import robust_ingestor, versioned_ingestor
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.runtime_post_index_publication_contract import install as install_post_index_contract
from rag_project.runtime_ready_publication_fix import install as install_ready_guards
from rag_project.storage.vector_store import VectorStore


WORKERS = 12


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


def _make_pdf(path: Path, pages: int = 1, text_prefix: str = "Publication phase regression") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    try:
        for page_number in range(1, pages + 1):
            page = document.new_page()
            page.insert_text(
                (72, 72),
                f"{text_prefix} page {page_number}. "
                "Stable searchable medical source material for post-index publication tests.",
            )
        document.save(str(path))
    finally:
        document.close()


def _make_system(settings):
    from rag_project.app.rag_system import RAGSystem

    return RAGSystem(settings)


def _make_vector_store(root: Path) -> VectorStore:
    install_post_index_contract()
    return VectorStore(root / "vectors")


def _seed_index(store: VectorStore, document_id: str = "doc", version_id: str = "v1", count: int = 3) -> None:
    documents = [f"medical source chunk {index}" for index in range(count)]
    ids = [f"{document_id}-{version_id}-{index}" for index in range(count)]
    metadatas = [
        {
            "document_id": document_id,
            "chunk_id": ids[index],
            "version_id": version_id,
            "page_numbers": [index + 1],
            "index_state": "BUILDING",
        }
        for index in range(count)
    ]
    embeddings = [[float(index + 1), 0.5, 0.25, 0.125] for index in range(count)]
    store.add_documents(documents, metadatas, embeddings, ids)


def _assert_final_ready(system, result):
    assert str(result.get("status", "")).upper() in {"SUCCESS", "READY", "COMPLETED"}, result
    document_id = str(result["document_id"])
    row = system.state_store.get_document(document_id)
    assert row is not None
    assert row["status"] == "READY"
    assert row["current_stage"] == "READY"
    assert row["index_state"] == "READY"
    assert int(row["current_page"]) == int(row["total_pages"]) > 0
    assert not row["lease_owner"]
    assert not row["lease_expires_at"]
    processed = Path(row["file_path"])
    assert processed.is_file()
    validation = system.vector_store.validate_document_index(document_id, row["version_id"])
    assert validation["valid"] is True, validation
    assert validation["semantic_count"] == validation["lexical_count"] == int(result["embedding_count"])
    return row


# ---------------------------------------------------------------------------
# Phase 6: READY publication
# ---------------------------------------------------------------------------


def test_phase6_ready_state_requires_semantic_lexical_parity(tmp_path):
    store = _make_vector_store(tmp_path)
    _seed_index(store)

    with sqlite3.connect(store.lexical_database) as db:
        db.execute("DELETE FROM lexical_documents WHERE id = ?", ("doc-v1-2",))
        db.commit()

    validation = store.validate_document_index("doc", "v1")
    assert validation["valid"] is False
    assert validation["semantic_count"] == 3
    assert validation["lexical_count"] == 2
    assert any("semantic/lexical count mismatch" in issue for issue in validation["issues"])


def test_phase6_ready_state_cannot_publish_when_lexical_chunk_is_missing(tmp_path):
    store = _make_vector_store(tmp_path)
    _seed_index(store)
    with sqlite3.connect(store.lexical_database) as db:
        db.execute("DELETE FROM lexical_documents WHERE id = ?", ("doc-v1-1",))
        db.commit()

    with pytest.raises(RuntimeError, match="READY publication contract"):
        store.set_version_index_state("doc", "v1", "READY")


def test_phase6_ready_state_publishes_both_indexes_atomically_enough_for_validation(tmp_path):
    store = _make_vector_store(tmp_path)
    _seed_index(store)
    store.set_version_index_state("doc", "v1", "READY")

    validation = store.validate_document_index("doc", "v1")
    assert validation["valid"] is True, validation
    assert validation["semantic_count"] == 3
    assert validation["lexical_count"] == 3
    with sqlite3.connect(store.lexical_database) as db:
        states = {row[0] for row in db.execute("SELECT index_state FROM lexical_documents")}
    assert states == {"READY"}


def test_phase6_ready_publication_is_idempotent(tmp_path):
    store = _make_vector_store(tmp_path)
    _seed_index(store)
    store.set_version_index_state("doc", "v1", "READY")
    store.set_version_index_state("doc", "v1", "READY")
    validation = store.validate_document_index("doc", "v1")
    assert validation["valid"] is True
    assert validation["semantic_count"] == validation["lexical_count"] == 3


def test_phase6_ready_document_survives_completion_audit_failure(tmp_path, monkeypatch):
    install_ready_guards()
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "audit-failure.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    original_record_event = system.state_store.record_event

    def fail_only_completion(document_id, **kwargs):
        if kwargs.get("event_type") == "completion":
            raise RuntimeError("injected completion audit failure")
        return original_record_event(document_id, **kwargs)

    monkeypatch.setattr(system.state_store, "record_event", fail_only_completion)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert row["status"] == "READY"


def test_phase6_ready_transition_rejects_incomplete_page_progress(tmp_path):
    install_ready_guards()
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    store.upsert_document(
        {
            "document_id": "doc",
            "content_hash": "hash",
            "file_path": str(tmp_path / "doc.pdf"),
            "file_name": "doc.pdf",
            "file_size": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "modified_at": "2026-01-01T00:00:00+00:00",
            "ingestion_started_at": "2026-01-01T00:00:00+00:00",
            "current_stage": "VALIDATING_INDEX",
            "current_page": 2,
            "total_pages": 3,
            "status": "RUNNING",
            "parser_version": "pdf-extractor-v3",
            "ocr_config": "{}",
            "chunking_config": "{}",
            "embedding_model": "test",
            "index_state": "READY",
            "version_id": "hash",
        }
    )
    with pytest.raises(RuntimeError, match="complete page progress"):
        store.transition_document_state("doc", "READY", current_page=2, total_pages=3, index_state="READY", content_hash="hash")


def test_phase6_ready_transition_requires_content_hash(tmp_path):
    install_ready_guards()
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    store.upsert_document(
        {
            "document_id": "doc",
            "content_hash": "hash",
            "file_path": str(tmp_path / "doc.pdf"),
            "file_name": "doc.pdf",
            "file_size": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "modified_at": "2026-01-01T00:00:00+00:00",
            "ingestion_started_at": "2026-01-01T00:00:00+00:00",
            "current_stage": "VALIDATING_INDEX",
            "current_page": 3,
            "total_pages": 3,
            "status": "RUNNING",
            "parser_version": "pdf-extractor-v3",
            "ocr_config": "{}",
            "chunking_config": "{}",
            "embedding_model": "test",
            "index_state": "READY",
            "version_id": "hash",
        }
    )
    store.update_document("doc", content_hash="")
    with pytest.raises(RuntimeError, match="content_hash"):
        store.transition_document_state("doc", "READY", current_page=3, total_pages=3, index_state="READY")


# ---------------------------------------------------------------------------
# Phase 7: file publication
# ---------------------------------------------------------------------------


def test_phase7_successful_ingestion_publishes_pdf_to_processed(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "file-publication.pdf"
    _make_pdf(source)
    system = _make_system(settings)

    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert not source.exists()
    assert Path(row["file_path"]).parent == settings.processed_dir.resolve()


def test_phase7_file_move_failure_never_returns_ready(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "move-failure.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    original_replace = Path.replace

    def fail_incoming_to_processed(self, target):
        target_path = Path(target)
        if self.resolve().parent == settings.incoming_dir.resolve() and target_path.resolve().parent == settings.processed_dir.resolve():
            raise OSError("injected processed publication move failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_incoming_to_processed)
    result = robust_ingestor.robust_ingest_file(system, source)
    assert str(result["status"]).lower() == "failed"
    row = system.state_store.get_document(result["document_id"])
    assert row is not None
    assert row["status"] != "READY"
    assert not (settings.processed_dir / source.name).exists()
    assert list(settings.failed_dir.glob("*.pdf")), "failed PDF must be quarantined"


def test_phase7_existing_processed_target_is_protected_until_new_publication_finishes(tmp_path):
    settings = _settings(tmp_path)
    existing_target = settings.processed_dir / "collision.pdf"
    _make_pdf(existing_target, text_prefix="existing processed target")
    source = settings.incoming_dir / "collision.pdf"
    _make_pdf(source, text_prefix="new incoming target")
    system = _make_system(settings)

    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert Path(row["file_path"]).read_bytes() != existing_target.read_bytes()
    assert existing_target.exists()


def test_phase7_file_publication_failure_restores_previous_processed_target(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    existing_target = settings.processed_dir / "restore.pdf"
    _make_pdf(existing_target, text_prefix="old processed target")
    source = settings.incoming_dir / "restore.pdf"
    _make_pdf(source, text_prefix="new incoming target")
    system = _make_system(settings)
    original_replace = Path.replace

    def fail_new_move(self, target):
        target_path = Path(target)
        if self.resolve() == source.resolve() and target_path.resolve() == existing_target.resolve():
            raise OSError("injected new move failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_new_move)
    result = robust_ingestor.robust_ingest_file(system, source)
    assert str(result["status"]).lower() == "failed"
    assert existing_target.exists()
    assert b"old processed target" in existing_target.read_bytes()


def test_phase7_archive_cleanup_failure_does_not_invalidate_ready_document(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    existing_target = settings.processed_dir / "cleanup.pdf"
    _make_pdf(existing_target, text_prefix="old processed target")
    source = settings.incoming_dir / "cleanup.pdf"
    _make_pdf(source, text_prefix="new incoming target")
    system = _make_system(settings)
    original_unlink = Path.unlink

    def fail_archive_unlink(self, *args, **kwargs):
        if self.parent.resolve() == settings.archive_dir.resolve():
            raise OSError("injected archive cleanup failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_archive_unlink)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert row["status"] == "READY"


# ---------------------------------------------------------------------------
# Phase 8: completion / audit event
# ---------------------------------------------------------------------------


def test_phase8_completion_event_is_recorded_on_success(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "completion-event.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    events = system.state_store.get_events(row["document_id"])
    assert any(event["event_type"] == "completion" and event["status"] == "READY" for event in events)


def test_phase8_completion_event_failure_is_non_destructive(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "completion-event-failure.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    original_record_event = system.state_store.record_event

    def completion_failure(document_id, **kwargs):
        if kwargs.get("event_type") == "completion":
            raise sqlite3.OperationalError("database busy during completion audit")
        return original_record_event(document_id, **kwargs)

    monkeypatch.setattr(system.state_store, "record_event", completion_failure)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert row["status"] == "READY"


def test_phase8_ready_event_is_after_durable_ready_transition(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "event-order.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    events = system.state_store.get_events(row["document_id"])
    ready_stage_indexes = [i for i, event in enumerate(events) if event["stage"] == "READY"]
    completion_indexes = [i for i, event in enumerate(events) if event["event_type"] == "completion"]
    assert ready_stage_indexes
    assert completion_indexes
    assert max(ready_stage_indexes) <= min(completion_indexes)


def test_phase8_event_details_contain_final_counts(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "event-details.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    completion = [event for event in system.state_store.get_events(row["document_id"]) if event["event_type"] == "completion"][-1]
    details = completion["details"]
    assert int(details["embedding_count"]) == int(result["embedding_count"])
    assert int(details["chunk_count"]) == int(result["chunk_count"])
    assert int(details["page_count"]) == int(result["page_count"])


# ---------------------------------------------------------------------------
# Phase 9: previous-version retirement
# ---------------------------------------------------------------------------


def test_phase9_changed_incoming_filename_retires_previous_ready_version(tmp_path):
    settings = _settings(tmp_path)
    system = _make_system(settings)
    first = settings.incoming_dir / "versioned.pdf"
    _make_pdf(first, text_prefix="version one")
    first_result = versioned_ingestor.ingest_version_safely(system, first)
    first_id = str(first_result["document_id"])
    old_hash = str(system.state_store.get_document(first_id)["content_hash"])

    second = settings.incoming_dir / "versioned.pdf"
    _make_pdf(second, text_prefix="version two changed content")
    second_result = versioned_ingestor.ingest_version_safely(system, second)

    assert second_result["status"] == "READY", second_result
    assert second_result["versioned_replacement"] is True
    assert second_result["previous_document_id"] == first_id
    assert second_result["previous_version_retired"] is True
    previous = system.state_store.get_document(first_id)
    assert previous is not None
    assert previous["status"] == "SUPERSEDED"
    assert previous["index_state"] == "FAILED"
    new_row = system.state_store.get_document(second_result["document_id"])
    assert new_row["status"] == "READY"
    validation = system.vector_store.validate_document_index(second_result["document_id"], new_row["content_hash"])
    assert validation["valid"] is True
    all_new_records = system.vector_store.get_documents(where={"document_id": second_result["document_id"]})
    assert not any(str(meta.get("version_id")) == old_hash for meta in all_new_records.get("metadatas", []))


def test_phase9_retirement_delete_failure_does_not_break_new_ready_version(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    system = _make_system(settings)
    first = settings.incoming_dir / "retirement-warning.pdf"
    _make_pdf(first, text_prefix="retirement old")
    first_result = versioned_ingestor.ingest_version_safely(system, first)
    old_id = str(first_result["document_id"])
    original_delete = system.vector_store.delete_version

    second = settings.incoming_dir / "retirement-warning.pdf"
    _make_pdf(second, text_prefix="retirement new")
    old_doc = system.state_store.get_document(old_id)
    assert old_doc

    def fail_only_old(document_id, version_id):
        if str(document_id) == old_id and str(version_id) == str(old_doc["content_hash"]):
            raise RuntimeError("injected old-version delete failure")
        return original_delete(document_id, version_id)

    monkeypatch.setattr(system.vector_store, "delete_version", fail_only_old)
    result = versioned_ingestor.ingest_version_safely(system, second)
    assert result["status"] == "READY"
    assert result["previous_version_retired"] is False
    assert result["previous_version_retirement_warnings"]
    new_row = system.state_store.get_document(result["document_id"])
    assert new_row["status"] == "READY"
    validation = system.vector_store.validate_document_index(result["document_id"], new_row["content_hash"])
    assert validation["valid"] is True


def test_phase9_retirement_event_failure_is_only_a_warning(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    system = _make_system(settings)
    first = settings.incoming_dir / "retirement-event.pdf"
    _make_pdf(first, text_prefix="event old")
    first_result = versioned_ingestor.ingest_version_safely(system, first)
    old_id = str(first_result["document_id"])
    second = settings.incoming_dir / "retirement-event.pdf"
    _make_pdf(second, text_prefix="event new")
    original_event = system.state_store.record_event

    def fail_retirement_event(document_id, **kwargs):
        if str(document_id) == old_id and kwargs.get("event_type") == "version_retired":
            raise RuntimeError("injected retirement audit failure")
        return original_event(document_id, **kwargs)

    monkeypatch.setattr(system.state_store, "record_event", fail_retirement_event)
    result = versioned_ingestor.ingest_version_safely(system, second)
    assert result["status"] == "READY"
    assert result["warning"]
    assert system.state_store.get_document(result["document_id"])["status"] == "READY"


def test_phase9_old_version_pages_are_removed_on_successful_retirement(tmp_path):
    settings = _settings(tmp_path)
    system = _make_system(settings)
    first = settings.incoming_dir / "retirement-pages.pdf"
    _make_pdf(first, pages=2, text_prefix="pages old")
    first_result = versioned_ingestor.ingest_version_safely(system, first)
    old_id = str(first_result["document_id"])
    assert system.state_store.get_pages(old_id)

    second = settings.incoming_dir / "retirement-pages.pdf"
    _make_pdf(second, pages=3, text_prefix="pages new")
    second_result = versioned_ingestor.ingest_version_safely(system, second)
    assert second_result["status"] == "READY"
    assert system.state_store.get_pages(old_id) == []


def test_phase9_only_new_version_remains_retrievable_as_ready(tmp_path):
    settings = _settings(tmp_path)
    system = _make_system(settings)
    source = settings.incoming_dir / "ready-version-only.pdf"
    _make_pdf(source, text_prefix="first")
    first = versioned_ingestor.ingest_version_safely(system, source)
    old_id = first["document_id"]
    source = settings.incoming_dir / "ready-version-only.pdf"
    _make_pdf(source, text_prefix="second")
    second = versioned_ingestor.ingest_version_safely(system, source)

    docs = system.state_store.get_all_documents()
    ready = [doc for doc in docs if system.state_store.is_ready_status(doc["status"])]
    assert len(ready) == 1
    assert ready[0]["document_id"] == second["document_id"]
    assert old_id != second["document_id"]


# ---------------------------------------------------------------------------
# Phase 10: lease / worker cleanup
# ---------------------------------------------------------------------------


def test_phase10_release_clears_lease(tmp_path):
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    values = {
        "document_id": "lease-doc",
        "content_hash": "lease-hash",
        "file_path": str(tmp_path / "lease.pdf"),
        "file_name": "lease.pdf",
        "file_size": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "modified_at": "2026-01-01T00:00:00+00:00",
        "ingestion_started_at": "2026-01-01T00:00:00+00:00",
        "current_stage": "DISCOVERED",
        "current_page": 0,
        "total_pages": 0,
        "status": "RUNNING",
        "parser_version": "pdf-extractor-v3",
        "ocr_config": "{}",
        "chunking_config": "{}",
        "embedding_model": "test",
        "index_state": "PENDING",
        "version_id": "lease-hash",
    }
    assert ensure_and_claim(store, values, "worker-1", 60)
    assert store.release_document("lease-doc", "worker-1") is True
    row = store.get_document("lease-doc")
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None
    assert row["heartbeat_at"] is None


def test_phase10_release_retries_transient_sqlite_failure(tmp_path, monkeypatch):
    install_ready_guards()
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    values = {
        "document_id": "lease-retry",
        "content_hash": "lease-retry-hash",
        "file_path": str(tmp_path / "lease-retry.pdf"),
        "file_name": "lease-retry.pdf",
        "file_size": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "modified_at": "2026-01-01T00:00:00+00:00",
        "ingestion_started_at": "2026-01-01T00:00:00+00:00",
        "current_stage": "DISCOVERED",
        "current_page": 0,
        "total_pages": 0,
        "status": "RUNNING",
        "parser_version": "pdf-extractor-v3",
        "ocr_config": "{}",
        "chunking_config": "{}",
        "embedding_model": "test",
        "index_state": "PENDING",
        "version_id": "lease-retry-hash",
    }
    assert ensure_and_claim(store, values, "worker-1", 60)
    original_connect = store._connect
    calls = {"count": 0}

    def flaky_connect():
        calls["count"] += 1
        if calls["count"] < 3:
            raise sqlite3.OperationalError("transient busy")
        return original_connect()

    monkeypatch.setattr(store, "_connect", flaky_connect)
    assert store.release_document("lease-retry", "worker-1") is True
    assert calls["count"] == 3


def test_phase10_wrong_worker_cannot_release_another_workers_lease(tmp_path):
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    values = {
        "document_id": "lease-owner",
        "content_hash": "lease-owner-hash",
        "file_path": str(tmp_path / "lease-owner.pdf"),
        "file_name": "lease-owner.pdf",
        "file_size": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "modified_at": "2026-01-01T00:00:00+00:00",
        "ingestion_started_at": "2026-01-01T00:00:00+00:00",
        "current_stage": "DISCOVERED",
        "current_page": 0,
        "total_pages": 0,
        "status": "RUNNING",
        "parser_version": "pdf-extractor-v3",
        "ocr_config": "{}",
        "chunking_config": "{}",
        "embedding_model": "test",
        "index_state": "PENDING",
        "version_id": "lease-owner-hash",
    }
    assert ensure_and_claim(store, values, "owner", 60)
    assert store.release_document("lease-owner", "intruder") is False
    assert store.get_document("lease-owner")["lease_owner"] == "owner"


def test_phase10_twelve_workers_have_single_atomic_claim_winner(tmp_path):
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    base = {
        "document_id": "twelve-worker-doc",
        "content_hash": "twelve-worker-hash",
        "file_path": str(tmp_path / "twelve-worker.pdf"),
        "file_name": "twelve-worker.pdf",
        "file_size": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "modified_at": "2026-01-01T00:00:00+00:00",
        "ingestion_started_at": "2026-01-01T00:00:00+00:00",
        "current_stage": "DISCOVERED",
        "current_page": 0,
        "total_pages": 0,
        "status": "RUNNING",
        "parser_version": "pdf-extractor-v3",
        "ocr_config": "{}",
        "chunking_config": "{}",
        "embedding_model": "test",
        "index_state": "PENDING",
        "version_id": "twelve-worker-hash",
    }

    def claim(worker_number):
        return ensure_and_claim(store, dict(base), f"worker-{worker_number}", 60)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        outcomes = list(pool.map(claim, range(WORKERS)))
    assert sum(bool(value) for value in outcomes) == 1
    row = store.get_document("twelve-worker-doc")
    assert row["lease_owner"] in {f"worker-{index}" for index in range(WORKERS)}


# ---------------------------------------------------------------------------
# Phase 11: final success invariants
# ---------------------------------------------------------------------------


def test_phase11_final_success_has_no_failed_or_building_chunks(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "final-invariants.pdf"
    _make_pdf(source, pages=4)
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    records = system.vector_store.get_documents(where={"document_id": row["document_id"]})
    states = {str(meta.get("index_state", "")).upper() for meta in records.get("metadatas", [])}
    assert states == {"READY"}


def test_phase11_final_success_has_exact_semantic_lexical_chunk_set(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "chunk-set.pdf"
    _make_pdf(source, pages=3)
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    records = system.vector_store.get_documents(where={"document_id": row["document_id"]})
    semantic_ids = {str(meta["chunk_id"]) for meta in records.get("metadatas", [])}
    with sqlite3.connect(system.vector_store.lexical_database) as db:
        lexical_ids = {
            str(row_id)
            for (row_id,) in db.execute("SELECT id FROM lexical_documents WHERE json_extract(metadata, '$.document_id') = ? AND upper(index_state) = 'READY'", (row["document_id"],))
        }
    assert semantic_ids == lexical_ids


def test_phase11_final_success_has_no_active_lease(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "lease-final.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert not row["lease_owner"]
    assert not row["lease_expires_at"]
    assert not row["heartbeat_at"]


def test_phase11_final_success_has_one_processed_file_and_no_failed_copy(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "single-final-file.pdf"
    _make_pdf(source)
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert list(settings.processed_dir.glob("*.pdf"))
    assert not list(settings.failed_dir.glob("*.pdf"))
    assert Path(row["file_path"]).name == source.name


def test_phase11_full_112_page_success_stays_ready(tmp_path):
    settings = _settings(tmp_path)
    source = settings.incoming_dir / "final-112-page.pdf"
    _make_pdf(source, pages=112, text_prefix="112-page final boundary")
    system = _make_system(settings)
    result = robust_ingestor.robust_ingest_file(system, source)
    row = _assert_final_ready(system, result)
    assert int(result["page_count"]) == 112
    assert int(row["current_page"]) == 112
    assert int(row["total_pages"]) == 112


# ---------------------------------------------------------------------------
# Cross-phase failure containment and 12-worker execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure_stage",
    ["ready_vector", "ready_document", "completion_event", "retirement_delete", "lease_release"],
)
def test_cross_phase_failures_never_corrupt_an_already_published_ready_version(tmp_path, monkeypatch, failure_stage):
    settings = _settings(tmp_path)
    system = _make_system(settings)
    source = settings.incoming_dir / f"cross-{failure_stage}.pdf"
    _make_pdf(source)

    if failure_stage == "ready_vector":
        original = system.vector_store.set_version_index_state
        monkeypatch.setattr(
            system.vector_store,
            "set_version_index_state",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected vector READY failure")),
        )
        result = robust_ingestor.robust_ingest_file(system, source)
        assert str(result["status"]).lower() == "failed"
        assert original
        return

    if failure_stage == "ready_document":
        original = system.state_store.transition_document_state

        def fail_ready(document_id, stage, **values):
            if str(stage).upper() == "READY":
                raise RuntimeError("injected durable READY transition failure")
            return original(document_id, stage, **values)

        monkeypatch.setattr(system.state_store, "transition_document_state", fail_ready)
        result = robust_ingestor.robust_ingest_file(system, source)
        assert str(result["status"]).lower() == "failed"
        row = system.state_store.get_document(result["document_id"])
        assert row["status"] != "READY"
        return

    if failure_stage == "completion_event":
        original = system.state_store.record_event

        def fail_completion(document_id, **kwargs):
            if kwargs.get("event_type") == "completion":
                raise RuntimeError("injected completion event failure")
            return original(document_id, **kwargs)

        monkeypatch.setattr(system.state_store, "record_event", fail_completion)
        result = robust_ingestor.robust_ingest_file(system, source)
        _assert_final_ready(system, result)
        return

    if failure_stage == "lease_release":
        original_release = system.state_store.release_document
        monkeypatch.setattr(system.state_store, "release_document", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected release failure")))
        result = robust_ingestor.robust_ingest_file(system, source)
        assert str(result["status"]).upper() == "SUCCESS"
        monkeypatch.setattr(system.state_store, "release_document", original_release)
        row = system.state_store.get_document(result["document_id"])
        assert row["status"] == "READY"
        return

    old = settings.processed_dir / source.name
    _make_pdf(old, text_prefix="old version")
    old_source = settings.incoming_dir / source.name
    _make_pdf(old_source, text_prefix="first version")
    first = versioned_ingestor.ingest_version_safely(system, old_source)
    old_id = str(first["document_id"])
    new_source = settings.incoming_dir / source.name
    _make_pdf(new_source, text_prefix="new version")
    previous = system.state_store.get_document(old_id)
    original_delete = system.vector_store.delete_version

    def fail_old_delete(document_id, version_id):
        if str(document_id) == old_id and str(version_id) == str(previous["content_hash"]):
            raise RuntimeError("injected retirement delete")
        return original_delete(document_id, version_id)

    monkeypatch.setattr(system.vector_store, "delete_version", fail_old_delete)
    result = versioned_ingestor.ingest_version_safely(system, new_source)
    assert result["status"] == "READY"
    assert system.state_store.get_document(result["document_id"])["status"] == "READY"


def test_cross_phase_12_worker_publication_claim_and_release_cycle(tmp_path):
    store = IngestionStateStore(tmp_path / "cycle.sqlite3")
    base = {
        "document_id": "cycle-doc",
        "content_hash": "cycle-hash",
        "file_path": str(tmp_path / "cycle.pdf"),
        "file_name": "cycle.pdf",
        "file_size": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "modified_at": "2026-01-01T00:00:00+00:00",
        "ingestion_started_at": "2026-01-01T00:00:00+00:00",
        "current_stage": "DISCOVERED",
        "current_page": 0,
        "total_pages": 0,
        "status": "RUNNING",
        "parser_version": "pdf-extractor-v3",
        "ocr_config": "{}",
        "chunking_config": "{}",
        "embedding_model": "test",
        "index_state": "PENDING",
        "version_id": "cycle-hash",
    }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda index: ensure_and_claim(store, dict(base), f"cycle-{index}", 60), range(WORKERS)))
    assert sum(results) == 1
    winner = store.get_document("cycle-doc")["lease_owner"]
    assert winner.startswith("cycle-")
    assert store.release_document("cycle-doc", winner) is True
    assert ensure_and_claim(store, dict(base), "replacement-worker", 60) is True
    assert store.release_document("cycle-doc", "replacement-worker") is True
