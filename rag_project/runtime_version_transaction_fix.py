from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return

        from rag_project.app.rag_system import RAGSystem

        original = getattr(RAGSystem, "ingest_file", None)
        if not callable(original) or getattr(original, "_runtime_version_transaction_fix", False):
            _INSTALLED = True
            return

        def ingest_file(self: Any, pdf_path: str | Path, *args: Any, **kwargs: Any):
            source = Path(pdf_path)
            vector_store = getattr(self, "vector_store", None)
            original_delete = getattr(vector_store, "delete_version", None)
            deferred: list[tuple[str, str]] = []

            # Read the currently published state before the replacement mutates
            # the document row. The old READY version is protected from every
            # internal delete_version call until the new version is successfully
            # published.
            try:
                previous = self.state_store.get_by_path(str(source.resolve()))
            except Exception:
                previous = None

            document_id = str((previous or {}).get("document_id") or "")
            protected_versions = {
                str((previous or {}).get("version_id") or ""),
                str((previous or {}).get("content_hash") or ""),
            }
            protected_versions.discard("")

            if vector_store is not None and callable(original_delete) and document_id and protected_versions:
                def guarded_delete(store: Any, target_document_id: str, target_version_id: str) -> None:
                    pair = (str(target_document_id), str(target_version_id))
                    if pair[0] == document_id and pair[1] in protected_versions:
                        if pair not in deferred:
                            deferred.append(pair)
                        return
                    return original_delete(target_document_id, target_version_id)

                vector_store.delete_version = guarded_delete.__get__(vector_store, type(vector_store))

            try:
                result = original(self, source, *args, **kwargs)
            finally:
                if vector_store is not None and callable(original_delete):
                    vector_store.delete_version = original_delete

            status = str(result.get("status") or "").casefold() if isinstance(result, dict) else ""
            if status in {"success", "ready", "completed", "skipped"}:
                # New READY publication succeeded. Retire the old version only now.
                for target_document_id, target_version_id in deferred:
                    try:
                        original_delete(target_document_id, target_version_id)
                    except Exception:
                        try:
                            self.logger.exception(
                                "Could not retire superseded version %s for %s",
                                target_version_id,
                                target_document_id,
                            )
                        except Exception:
                            pass
            # Any failed replacement leaves deferred old records untouched.
            return result

        ingest_file._runtime_version_transaction_fix = True
        ingest_file.__name__ = getattr(original, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original, "__qualname__", "ingest_file")
        RAGSystem.ingest_file = ingest_file
        _INSTALLED = True


__all__ = ["install"]
