from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rag_project.configuration.settings import Settings
from rag_project.embeddings.embedding_service import EmbeddingService
from rag_project.storage.vector_store import VectorStore


def rebuild_index(project_root: str | Path | None = None) -> dict[str, Any]:
    """Build a validated replacement index without modifying the active index."""
    root = Path(project_root).resolve() if project_root else Path.cwd()
    settings = Settings(project_root=root)
    source = VectorStore(settings.vector_db_dir)
    records = source.collection.get(include=["documents", "metadatas"])
    documents = [str(value) for value in records.get("documents", [])]
    ids = [str(value) for value in records.get("ids", [])]
    metadatas = []
    for item_id, value in zip(ids, records.get("metadatas", []), strict=True):
        metadata = dict(value or {})
        metadata.setdefault("chunk_id", item_id)
        metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
        metadata.setdefault("page_numbers", [])
        metadata.setdefault("index_state", "READY")
        metadatas.append(metadata)
    if not documents:
        raise RuntimeError("Cannot rebuild an empty index.")

    service = EmbeddingService(
        settings.ollama_base_url,
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        retries=settings.embedding_retries,
        test_mode=settings.embedding_test_mode,
        cache_size=0,
    )
    service.discover_dimension()
    staging = settings.vector_db_dir.parent / (
        f"{settings.vector_db_dir.name}.staging-{time.strftime('%Y%m%d%H%M%S')}"
    )
    replacement = VectorStore(staging)
    replacement.set_expected_identity(service.identity)
    embeddings = service.embed_texts(documents)
    replacement.add_documents(documents, metadatas, embeddings, ids)
    health = replacement.index_health_check(service.identity)
    if not health["valid"] or not health["metadata_valid"]:
        raise RuntimeError(json.dumps(health, default=str))
    return {
        "status": "STAGED",
        "staging_directory": str(staging),
        "active_directory": str(settings.vector_db_dir),
        "vector_count": len(documents),
        "health": health,
        "message": "Replacement validated; active index was not changed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a validated, non-destructive replacement index.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    print(json.dumps(rebuild_index(args.project_root), indent=2, default=str))


if __name__ == "__main__":
    main()
