from __future__ import annotations

from pathlib import Path

import chromadb

from rag_project.ingestion.state_store import IngestionStateStore, utc_now
from rag_project.runtime_chroma_metadata_fix import install as install_metadata_fix
from rag_project.storage.vector_store import VectorStore


def _seed_document_state(path: Path) -> IngestionStateStore:
    store = IngestionStateStore(path)
    store.upsert_document(
        {
            "document_id": "doc-ready",
            "content_hash": "hash-ready",
            "file_path": str(path.parent / "doc.pdf"),
            "file_name": "doc.pdf",
            "file_size": 1,
            "created_at": utc_now(),
            "modified_at": utc_now(),
            "ingestion_started_at": utc_now(),
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
    return store


def test_ready_publication_sets_document_index_state_ready(tmp_path):
    install_metadata_fix()
    store = _seed_document_state(tmp_path / "ingestion.sqlite3")

    store.transition_document_state(
        "doc-ready",
        "READY",
        current_page=112,
        total_pages=112,
        content_hash="hash-ready",
    )

    document = store.get_document("doc-ready")
    assert document is not None
    assert document["status"] == "READY"
    assert document["current_stage"] == "READY"
    assert document["index_state"] == "READY"


def test_ready_publication_can_update_chroma_metadata_without_empty_list_failure(tmp_path):
    install_metadata_fix()
    store = VectorStore(tmp_path / "vectors")
    store.collection.add(
        ids=["chunk-0"],
        documents=["medical evidence"],
        metadatas=[
            {
                "document_id": "doc-ready",
                "version_id": "hash-ready",
                "chunk_id": "chunk-0",
            }
        ],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    store.set_version_index_state("doc-ready", "hash-ready", "READY")

    records = store.collection.get(include=["metadatas"])
    assert records["metadatas"]
    assert records["metadatas"][0]["index_state"] == "READY"
    assert "page_numbers" not in records["metadatas"][0]
