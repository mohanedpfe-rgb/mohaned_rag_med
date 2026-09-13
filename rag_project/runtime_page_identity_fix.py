from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _recover_document_id(store: Any, candidate: Any, source_path: Any = None) -> str:
    document_id = str(candidate or "").strip()
    if document_id:
        return document_id
    if source_path:
        try:
            recovered = store.get_by_path(str(Path(source_path).expanduser().resolve()))
        except Exception:
            recovered = None
        if recovered and recovered.get("document_id"):
            return str(recovered["document_id"])
    try:
        documents = store.get_all_documents()
    except Exception:
        documents = []
    if len(documents) == 1 and documents[0].get("document_id"):
        return str(documents[0]["document_id"])
    return ""


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.ingestion.state_store import IngestionStateStore

        original_record = getattr(IngestionStateStore, "record_page", None)
        original_upsert = getattr(IngestionStateStore, "upsert_page", None)

        if callable(original_upsert) and not getattr(original_upsert, "_runtime_page_identity_fix", False):
            def upsert_page(self: Any, document_id: str, page_number: int, **values: Any) -> None:
                resolved = _recover_document_id(self, document_id, values.get("source_path"))
                if not resolved:
                    raise ValueError("page checkpoint cannot be persisted without document_id")
                return original_upsert(self, resolved, page_number, **values)

            upsert_page._runtime_page_identity_fix = True
            upsert_page.__name__ = getattr(original_upsert, "__name__", "upsert_page")
            upsert_page.__qualname__ = f"{IngestionStateStore.__name__}.upsert_page"
            IngestionStateStore.upsert_page = upsert_page

        if callable(original_record) and not getattr(original_record, "_runtime_page_identity_fix", False):
            def record_page(self: Any, extraction: Any, *, cache_reference: str | None = None) -> None:
                document_id = _recover_document_id(self, getattr(extraction, "document_id", ""), getattr(extraction, "source_path", None))
                if not document_id:
                    raise ValueError("page checkpoint cannot be persisted without document_id")
                try:
                    setattr(extraction, "document_id", document_id)
                except Exception:
                    pass
                return original_record(self, extraction, cache_reference=cache_reference)

            record_page._runtime_page_identity_fix = True
            record_page.__name__ = getattr(original_record, "__name__", "record_page")
            record_page.__qualname__ = f"{IngestionStateStore.__name__}.record_page"
            IngestionStateStore.record_page = record_page

        _INSTALLED = True


__all__ = ["install"]
