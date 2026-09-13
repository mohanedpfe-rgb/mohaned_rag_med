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
        from rag_project.ingestion.state_store import IngestionStateStore

        original = getattr(IngestionStateStore, "record_page", None)
        if not callable(original) or getattr(original, "_runtime_page_identity_fix", False):
            _INSTALLED = True
            return

        def record_page(self: Any, extraction: Any, *, cache_reference: str | None = None) -> None:
            document_id = str(getattr(extraction, "document_id", "") or "").strip()
            if not document_id:
                source_path = getattr(extraction, "source_path", None)
                if source_path:
                    try:
                        recovered = self.get_by_path(str(Path(source_path).expanduser().resolve()))
                    except Exception:
                        recovered = None
                    if recovered and recovered.get("document_id"):
                        document_id = str(recovered["document_id"])
                        try:
                            setattr(extraction, "document_id", document_id)
                        except Exception:
                            pass
            if not document_id:
                raise ValueError(
                    "page checkpoint cannot be persisted without document_id; "
                    "no document identity could be recovered from source_path"
                )
            return original(self, extraction, cache_reference=cache_reference)

        record_page._runtime_page_identity_fix = True
        record_page.__name__ = getattr(original, "__name__", "record_page")
        record_page.__qualname__ = f"{IngestionStateStore.__name__}.record_page"
        IngestionStateStore.record_page = record_page
        _INSTALLED = True


__all__ = ["install"]
