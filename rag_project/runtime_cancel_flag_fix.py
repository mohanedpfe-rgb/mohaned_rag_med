from __future__ import annotations

import threading
from typing import Any


def install() -> None:
    """Restore lifecycle guards and neutralize unsafe dynamic lexical lookup patching."""
    from rag_project.app.rag_system import (
        RAGSystem,
        _INGEST_CANCEL_FLAGS,
        _INGEST_LOCK,
        _IngestCancelFlag,
    )

    if not getattr(RAGSystem, "_cancel_flag_contract_v1", False):
        def _new_cancel_flag(self: Any, document_id: str) -> _IngestCancelFlag:
            key = str(document_id)
            with _INGEST_LOCK:
                existing = _INGEST_CANCEL_FLAGS.get(key)
                if existing is None or existing.cancelled:
                    existing = _IngestCancelFlag()
                    _INGEST_CANCEL_FLAGS[key] = existing
                return existing

        def _remove_cancel_flag(self: Any, document_id: str) -> None:
            with _INGEST_LOCK:
                _INGEST_CANCEL_FLAGS.pop(str(document_id), None)

        def cancel_ingest(self: Any, document_id: str) -> bool:
            with _INGEST_LOCK:
                flag = _INGEST_CANCEL_FLAGS.get(str(document_id))
            if flag is None:
                return False
            flag.cancel()
            return True

        RAGSystem._new_cancel_flag = _new_cancel_flag
        RAGSystem._remove_cancel_flag = _remove_cancel_flag
        RAGSystem.cancel_ingest = cancel_ingest
        RAGSystem._cancel_flag_contract_v1 = True

    original_ingest = getattr(RAGSystem, "ingest_file", None)
    if callable(original_ingest) and not getattr(original_ingest, "_empty_pdf_status_normalizer", False):
        def ingest_file(self: Any, pdf_path: Any, *args: Any, **kwargs: Any):
            result = original_ingest(self, pdf_path, *args, **kwargs)
            if isinstance(result, dict) and str(result.get("status") or "").upper() == "FAILED":
                document_id = str(result.get("document_id") or result.get("id") or "")
                if document_id and getattr(self, "state_store", None) is not None:
                    row = self.state_store.get_document(document_id)
                    error = str((row or {}).get("error") or "")
                    if str((row or {}).get("status") or "").upper() == "FAILED_EXTRACTION" and "no extractable searchable content" in error.casefold():
                        self.state_store.update_document(
                            document_id,
                            current_stage="FAILED",
                            status="FAILED",
                            index_state="FAILED",
                        )
            return result

        ingest_file._empty_pdf_status_normalizer = True
        ingest_file.__name__ = getattr(original_ingest, "__name__", "ingest_file")
        ingest_file.__qualname__ = getattr(original_ingest, "__qualname__", ingest_file.__name__)
        RAGSystem.ingest_file = ingest_file

    try:
        from rag_project.storage.vector_store import VectorStore

        # runtime.py historically replaced __getattribute__ dynamically to
        # enforce READY-only lexical retrieval. That interception is unsafe:
        # repeated attribute access can wrap an already-wrapped method and lead
        # to recursive lookup. The static search wrapper already applies the
        # authoritative SQL index_state filter.
        if getattr(VectorStore, "_final_dynamic_boundary", False):
            VectorStore.__getattribute__ = object.__getattribute__

        current_search = VectorStore.search_lexical
        if not getattr(current_search, "_runtime_final_ready_state_guard", False):
            def search_lexical(self: Any, query: Any, n_results: int = 5, where: Any = None):
                # The historical wrapper cached blocked IDs in memory. Those
                # IDs can become READY later, so never trust that cache.
                # SQLite index_state is the single source of truth.
                try:
                    setattr(self, "_final_nonready_lexical_ids", set())
                except Exception:
                    pass
                return current_search(self, query, n_results=n_results, where=where)

            search_lexical._runtime_final_ready_state_guard = True
            VectorStore.search_lexical = search_lexical
    except Exception:
        pass


__all__ = ["install"]
