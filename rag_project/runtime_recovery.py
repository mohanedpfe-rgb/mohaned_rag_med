from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any


_BUILD_CONTEXT = threading.local()


def _set_build_context(system: Any, pdf_path: str | Path) -> None:
    path = Path(pdf_path)
    try:
        content_hash = system._hash_file(path)
    except Exception:
        content_hash = ""
    document = None
    if content_hash:
        document = system.state_store.get_by_hash(content_hash)
    if document is None:
        try:
            document = system.state_store.get_by_path(str(path.resolve()))
        except Exception:
            document = None
    _BUILD_CONTEXT.value = {
        "system": system,
        "document_id": document.get("document_id") if document else content_hash,
        "content_hash": content_hash,
    }


def _clear_build_context() -> None:
    try:
        del _BUILD_CONTEXT.value
    except AttributeError:
        pass


def _safe_embed_texts(self: Any, *args: Any, **kwargs: Any):
    """On a semantic failure, immediately remove BUILDING semantic/lexical records."""
    try:
        original = self._original_runtime_recovery_embed_texts
    except AttributeError:
        original = self._runtime_recovery_original_embed_texts
    try:
        return original(*args, **kwargs)
    except Exception:
        context = getattr(_BUILD_CONTEXT, "value", None)
        if context:
            system = context.get("system")
            document_id = context.get("document_id")
            version_id = context.get("content_hash")
            if system is not None and document_id and version_id:
                try:
                    system.vector_store.delete_version(document_id, version_id)
                except Exception:
                    system.logger.exception(
                        "Failed to clean semantic BUILDING records after embedding failure"
                    )
        raise


def _safe_ingest_file(self: Any, pdf_path: str | Path):
    original = self._original_runtime_recovery_ingest_file
    _set_build_context(self, pdf_path)
    try:
        return original(pdf_path)
    finally:
        _clear_build_context()


def install() -> None:
    from rag_project.app.rag_system import RAGSystem
    from rag_project.embeddings.embedding_service import EmbeddingService

    if not hasattr(EmbeddingService, "_original_runtime_recovery_embed_texts"):
        original = EmbeddingService.embed_texts
        EmbeddingService._original_runtime_recovery_embed_texts = original
        EmbeddingService._runtime_recovery_original_embed_texts = original
        EmbeddingService.embed_texts = _safe_embed_texts

    if not hasattr(RAGSystem, "_original_runtime_recovery_ingest_file"):
        RAGSystem._original_runtime_recovery_ingest_file = RAGSystem.ingest_file
        RAGSystem.ingest_file = _safe_ingest_file
