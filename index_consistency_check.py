from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_project.configuration.settings import Settings
from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.storage.vector_store import VectorStore


def run_index_consistency_check(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parent
    settings = Settings(project_root=root)
    state_store = IngestionStateStore(settings.ingestion_db_path)
    vector_store = VectorStore(settings.vector_db_dir)
    embedding_service = EmbeddingService(
        settings.ollama_base_url,
        settings.embedding_model,
        retries=settings.embedding_retries,
        test_mode=settings.embedding_test_mode,
        cache_size=0,
    )
    embedding_service.discover_dimension()
    vector_store.set_expected_identity(embedding_service.identity)

    documents = state_store.get_all_documents() if hasattr(state_store, "get_all_documents") else []
    if not documents:
        documents = []

    results: list[dict[str, Any]] = []
    for document in documents:
        document_id = document.get("document_id")
        if not document_id:
            continue
        vector_result = vector_store.validate_document_index(str(document_id))
        results.append({
            "document_id": document_id,
            "document_status": document.get("status"),
            "index_state": document.get("index_state"),
            "vector_valid": vector_result.get("valid", False),
            "issues": vector_result.get("issues", []),
            "vector_count": vector_result.get("count", 0),
        })

    health = vector_store.index_health_check(embedding_service.identity)
    return {
        "project_root": str(root),
        "documents": results,
        "vector_collection_count": vector_store.count(),
        "expected_dimension": health["expected_dimension"],
        "collection_dimension": health["collection_dimension"],
        "embedding_identity": health["expected_identity"],
        "stored_embedding_identity": health["stored_identity"],
        "metadata_issues": health["metadata_issues"],
        "valid": health["valid"] and health["metadata_valid"] and all(
            item["vector_valid"] for item in results
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify vector store and SQLite document state consistency.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root used to resolve config and state directories.")
    args = parser.parse_args()
    report = run_index_consistency_check(args.project_root)
    print(json.dumps(report, indent=2, default=str))
