from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Sequence


def _safe_add_documents(
    self: Any,
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    ids: Sequence[str],
) -> None:
    """Two-phase-ish vector/lexical write: upsert vectors, then lexical, with rollback."""
    if not documents:
        return
    if not (len(documents) == len(metadatas) == len(embeddings) == len(ids)):
        raise ValueError("documents, metadatas, embeddings, and ids must have the same length")
    lock = getattr(self, "_transaction_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._transaction_lock = lock
    with lock:
        dim = self._resolve_dimension(embeddings)
        self._apply_collection_metadata(dim)
        normalized = []
        for index, item in enumerate(documents):
            metadata = self._coerce_metadata(metadatas[index])
            metadata.setdefault("document_id", "unknown")
            metadata.setdefault("chunk_id", str(ids[index]))
            metadata.setdefault("index_state", "BUILDING")
            metadata.setdefault("version_id", metadata.get("document_id", "legacy"))
            metadata.setdefault("page_numbers", [])
            if not self._valid_vector(embeddings[index], dim):
                raise ValueError(f"Invalid semantic embedding at index {index}.")
            normalized.append(metadata)

        safe_ids = [str(item) for item in ids]
        try:
            # Upsert makes retries and deterministic chunk IDs idempotent.
            self.collection.upsert(
                ids=safe_ids,
                documents=[str(item) for item in documents],
                metadatas=normalized,
                embeddings=[list(map(float, vector)) for vector in embeddings],
            )
            self._update_collection_identity(self.expected_identity)
            try:
                self._upsert_lexical_records(documents, normalized, safe_ids)
            except Exception:
                # BUILDING vectors are not retrievable, but remove them to avoid
                # indefinite orphan records when the lexical transaction fails.
                try:
                    self.collection.delete(ids=safe_ids)
                except Exception:
                    pass
                raise
        except Exception:
            raise


def _safe_ingestion_version_id(
    *,
    content_hash: str,
    parser_version: str,
    ocr_config: str | dict[str, Any] | None,
    chunking_config: str | dict[str, Any] | None,
    embedding_model: str,
    embedding_profile: str | None,
    embedding_dimension: int | None,
) -> str:
    """Stable version fingerprint that also captures Ollama endpoint and declared dimension."""
    def as_dict(value: str | dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"value": value}
        return value or {}

    dimension = embedding_dimension
    if not dimension:
        try:
            env_dim = int(os.getenv("EMBEDDING_DIMENSION", "0"))
            dimension = env_dim or None
        except ValueError:
            dimension = None
    endpoint = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    profile = embedding_profile or hashlib.sha256(
        f"ollama|{endpoint}|{embedding_model}|{dimension or 'unknown'}".encode("utf-8")
    ).hexdigest()
    payload = {
        "content_hash": content_hash,
        "parser_version": parser_version,
        "ocr_config": as_dict(ocr_config),
        "chunking_config": as_dict(chunking_config),
        "embedding_model": embedding_model,
        "embedding_profile": profile,
        "embedding_dimension": dimension,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def install() -> None:
    from rag_project.storage.vector_store import VectorStore
    from rag_project.app.rag_system import RAGSystem

    if not hasattr(VectorStore, "_original_safe_add_documents"):
        VectorStore._original_safe_add_documents = VectorStore.add_documents
        VectorStore.add_documents = _safe_add_documents

    if not hasattr(RAGSystem, "_original_safe_ingestion_version_id"):
        RAGSystem._original_safe_ingestion_version_id = RAGSystem._ingestion_version_id
        RAGSystem._ingestion_version_id = staticmethod(_safe_ingestion_version_id)
