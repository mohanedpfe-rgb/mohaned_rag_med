from __future__ import annotations

import concurrent.futures
import inspect
import json
import sqlite3
from pathlib import Path

import fitz
import pytest

from rag_project.app.rag_system import RAGSystem
from rag_project.configuration.settings import Settings
from rag_project.storage.vector_store import VectorStore


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
    )


def _write_pdf(path: Path, *pages: str) -> None:
    document = fitz.open()
    try:
        for text in pages:
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(str(path))
    finally:
        document.close()


def test_full_pdf_transaction_reaches_ready_and_is_retrievable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.incoming_dir.mkdir(parents=True)
    pdf_path = settings.incoming_dir / "medical.pdf"
    _write_pdf(
        pdf_path,
        "Medical source text: acute appendicitis commonly presents with right lower quadrant pain.",
        "Medical source text: fever and nausea may accompany the clinical presentation.",
    )

    system = RAGSystem(settings)
    result = system.ingest_file(pdf_path)

    assert result["status"] == "success", result
    assert result["embedding_count"] > 0
    document = system.state_store.get_document(result["document_id"])
    assert document is not None
    assert document["status"] == "READY"
    assert document["index_state"] == "READY"

    version_id = str(document.get("version_id") or "")
    assert version_id

    print("========== DEBUG VALIDATION START ==========")
    print("DEBUG TEST VECTOR_STORE TYPE:", type(system.vector_store))
    print(
        "DEBUG TEST VECTOR_STORE FILE:",
        inspect.getfile(type(system.vector_store)),
    )
    method = system.vector_store.validate_document_index
    print(
        "DEBUG TEST VECTOR_STORE METHOD:",
        method.__func__.__code__.co_filename,
    )
    print(
        "DEBUG TEST VECTOR_STORE METHOD NAME:",
        method.__func__.__name__,
    )
    print(
        "DEBUG TEST VECTOR_STORE OBJECT ID:",
        id(system.vector_store),
    )
    print("DEBUG TEST DOCUMENT ID:", result["document_id"])
    print("DEBUG TEST VERSION ID:", version_id)
    print(
        "DEBUG TEST PERSIST DIRECTORY:",
        getattr(system.vector_store, "persist_directory", "MISSING"),
    )
    print(
        "DEBUG TEST COLLECTION NAME:",
        getattr(system.vector_store, "collection_name", "MISSING"),
    )
    print(
        "DEBUG TEST COLLECTION OBJECT ID:",
        id(getattr(system.vector_store, "collection", None)),
    )

    raw_records = system.vector_store.collection.get(
        where={"document_id": result["document_id"]},
        include=["metadatas", "documents", "embeddings"],
    )
    raw_ids = list(raw_records.get("ids") or [])
    raw_metadatas = list(raw_records.get("metadatas") or [])
    raw_documents = list(raw_records.get("documents") or [])
    raw_embeddings = list(raw_records.get("embeddings") or [])
    print("DEBUG RAW RECORD COUNT:", len(raw_ids))
    print("DEBUG RAW IDS:", raw_ids)
    print("DEBUG RAW METADATA COUNT:", len(raw_metadatas))
    for index, metadata in enumerate(raw_metadatas):
        print(
            f"DEBUG RAW METADATA[{index}]:",
            metadata,
        )
        print(
            f"DEBUG RAW DOCUMENT_ID[{index}]:",
            metadata.get("document_id") if isinstance(metadata, dict) else None,
        )
        print(
            f"DEBUG RAW VERSION_ID[{index}]:",
            metadata.get("version_id") if isinstance(metadata, dict) else None,
        )
        print(
            f"DEBUG RAW INDEX_STATE[{index}]:",
            metadata.get("index_state") if isinstance(metadata, dict) else None,
        )
        print(
            f"DEBUG RAW CHUNK_ID[{index}]:",
            metadata.get("chunk_id") if isinstance(metadata, dict) else None,
        )
        print(
            f"DEBUG RAW EMBEDDING_DIM[{index}]:",
            len(raw_embeddings[index]) if index < len(raw_embeddings) and raw_embeddings[index] is not None else None,
        )
        print(
            f"DEBUG RAW DOCUMENT_TEXT_LENGTH[{index}]:",
            len(str(raw_documents[index])) if index < len(raw_documents) else None,
        )

    matching_version_indexes = [
        index
        for index, metadata in enumerate(raw_metadatas)
        if isinstance(metadata, dict)
        and str(metadata.get("version_id")) == version_id
    ]
    print("DEBUG EXPECTED VERSION MATCHING INDEXES:", matching_version_indexes)
    print("DEBUG EXPECTED VERSION MATCH COUNT:", len(matching_version_indexes))
    print("DEBUG RESULT EMBEDDING COUNT:", result["embedding_count"])
    print("DEBUG STATE DOCUMENT VERSION_ID:", document.get("version_id"))
    print("========== DEBUG VALIDATION END ==========")

    validation = system.vector_store.validate_document_index(
        result["document_id"], version_id
    )
    print("DEBUG VALIDATION RESULT:", validation)
    assert validation["valid"] is True, validation
    assert validation["count"] == result["embedding_count"]

    records = system.vector_store.get_documents(
        {"document_id": result["document_id"]}
    )
    assert records["documents"]
    assert any("appendicitis" in str(text).lower() for text in records["documents"])


def test_empty_vector_store_reports_unknown_dimension_without_fabrication(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vectors")
    assert store._resolve_dimension() == 0


def test_vector_store_lexical_writes_are_thread_safe(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vectors")
    records = [
        (
            [f"text-{index}"],
            [
                {
                    "document_id": "doc-concurrent",
                    "chunk_id": f"chunk-{index}",
                    "version_id": "v1",
                    "index_state": "READY",
                }
            ],
            [f"id-{index}"],
        )
        for index in range(24)
    ]

    def write(record: tuple[list[str], list[dict[str, object]], list[str]]) -> None:
        documents, metadata, ids = record
        store.add_lexical_documents(documents, metadata, ids)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, records))

    assert store.lexical_count() == 24
    with sqlite3.connect(store.lexical_database) as connection:
        rows = connection.execute(
            "SELECT id, metadata FROM lexical_documents ORDER BY id"
        ).fetchall()
    assert len(rows) == 24
    assert all(json.loads(metadata)["version_id"] == "v1" for _, metadata in rows)
