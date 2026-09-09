from __future__ import annotations

import json
import sqlite3

from rag_project.quality_gate import audit_index_consistency, repair_index_consistency, validate_runtime_contract
from rag_project.storage.vector_store import VectorStore


class _Settings:
    incoming_dir = None
    processed_dir = None
    failed_dir = None
    archive_dir = None
    vector_db_dir = None
    log_dir = None
    embedding_model = "test-model"
    ollama_base_url = "http://127.0.0.1:11434"


def test_runtime_contract_reports_missing_paths_and_vector_methods(tmp_path):
    class Incomplete:
        settings = _Settings()
        vector_store = object()

    system = Incomplete()
    system.settings.incoming_dir = tmp_path / "incoming"
    system.settings.processed_dir = tmp_path / "processed"
    system.settings.failed_dir = tmp_path / "failed"
    system.settings.archive_dir = tmp_path / "archive"
    system.settings.vector_db_dir = tmp_path / "vectors"
    system.settings.log_dir = tmp_path / "logs"
    report = validate_runtime_contract(system)
    assert report["ok"] is False
    assert "add_documents" in report["missing_vector_methods"]


def test_quality_repair_restores_missing_and_stale_lexical_rows(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    store.add_documents(
        ["alpha treatment"],
        [{"document_id": "doc-1", "chunk_id": "chunk-1", "version_id": "v1", "page_numbers": [1], "index_state": "READY"}],
        [[0.1, 0.2]],
        ["vec-1"],
    )

    with sqlite3.connect(store.lexical_database) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO lexical_documents(id, document, metadata, index_state, tokens) VALUES (?, ?, ?, ?, ?)",
            (
                "orphan",
                "stale record",
                json.dumps({"document_id": "doc-1", "version_id": "v1", "index_state": "READY"}),
                "READY",
                json.dumps(["stale", "record"]),
            ),
        )
        connection.execute("DELETE FROM lexical_documents WHERE id = ?", ("vec-1",))
        connection.commit()

    before = audit_index_consistency(type("S", (), {"vector_store": store})())
    assert before["valid"] is False

    result = repair_index_consistency(type("S", (), {"vector_store": store})())
    assert result["after"]["valid"] is True
    assert result["after"]["vector_count"] == 1
    assert result["after"]["lexical_count"] == 1


def test_repaired_lexical_content_matches_authoritative_vector_record(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    store.add_documents(
        ["original content"],
        [{"document_id": "doc-2", "chunk_id": "chunk-2", "version_id": "v2", "page_numbers": [2], "index_state": "READY"}],
        [[0.2, 0.3]],
        ["vec-2"],
    )
    with sqlite3.connect(store.lexical_database) as connection:
        connection.execute(
            "UPDATE lexical_documents SET document = ?, metadata = ? WHERE id = ?",
            (
                "tampered content",
                json.dumps({"document_id": "doc-2", "chunk_id": "chunk-2", "version_id": "wrong", "index_state": "READY"}),
                "vec-2",
            ),
        )
        connection.commit()

    system = type("S", (), {"vector_store": store})()
    result = repair_index_consistency(system, "doc-2")
    assert result["after"]["valid"] is True
    with sqlite3.connect(store.lexical_database) as connection:
        row = connection.execute("SELECT document, metadata FROM lexical_documents WHERE id = ?", ("vec-2",)).fetchone()
    assert row[0] == "original content"
    assert json.loads(row[1])["version_id"] == "v2"
