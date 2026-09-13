from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    """Make replacement ingestion atomic: build new version first, retire old only after READY."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.app.rag_system import RAGSystem

        original = getattr(RAGSystem, "ingest_file", None)
        if not callable(original) or getattr(original, "_transactional_rollback_fix", False):
            _INSTALLED = True
            return

        def ingest_file(self: Any, pdf_path: str | Path, *args: Any, **kwargs: Any):
            source = Path(pdf_path)
            deferred: list[tuple[str, str]] = []
            vector_store = getattr(self, "vector_store", None)
            original_delete = getattr(vector_store, "delete_version", None)

            # Capture the currently published version before the replacement starts.
            # The old version must remain searchable while the new version is being
            # extracted, embedded, validated, and activated.
            try:
                previous = self.state_store.get_by_path(str(source.resolve()))
            except Exception:
                previous = None

            previous_document_id = str((previous or {}).get("document_id") or "")
            protected_versions = {
                str((previous or {}).get("version_id") or ""),
                str((previous or {}).get("content_hash") or ""),
            }
            protected_versions.discard("")

            if vector_store is not None and callable(original_delete) and protected_versions:
                def deferred_delete(store: Any, document_id: str, version_id: str) -> None:
                    version_text = str(version_id)
                    if (
                        str(document_id) == previous_document_id
                        and version_text in protected_versions
                    ):
                        pair = (str(document_id), version_text)
                        if pair not in deferred:
                            deferred.append(pair)
                        return
                    return original_delete(document_id, version_id)

                vector_store.delete_version = deferred_delete.__get__(vector_store, type(vector_store))

            try:
                result = original(self, source, *args, **kwargs)
            finally:
                if vector_store is not None and callable(original_delete):
                    vector_store.delete_version = original_delete

            status = str(result.get("status") or "").casefold() if isinstance(result, dict) else ""
            if status in {"success", "ready", "completed", "skipped"} and deferred:
                # Publication succeeded. Only now is the old searchable version retired.
                for document_id, version_id in deferred:
                    try:
                        original_delete(document_id, version_id)
                    except Exception:
                        try:
                            self.logger.exception(
                                "Failed to retire superseded version %s for %s",
                                version_id,
                                document_id,
                            )
                        except Exception:
                            pass
            # Failed activation/indexing intentionally leaves the old version intact.
            return result

        ingest_file._transactional_rollback_fix = True
        ingest_file.__name__ = getattr(original, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original, "__qualname__", "ingest_file")
        RAGSystem.ingest_file = ingest_file
        _INSTALLED = True


__all__ = ["install"]
