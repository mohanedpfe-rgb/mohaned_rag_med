"""Final ingestion-state functionality corrections."""
from __future__ import annotations

from typing import Any


def _wrap_document_ready(original):
    def wrapped(self: Any, document_id: str) -> bool:
        record = self.get_document(document_id)
        if not record:
            return False
        status = str(record.get("status") or "").upper()
        index_state = str(record.get("index_state") or "").upper()
        return status in {"READY", "COMPLETED"} and index_state == "READY"

    wrapped._functionality_state_ready_guard = True
    return wrapped


def install() -> None:
    from rag_project.ingestion.state_store import IngestionStateStore

    original = IngestionStateStore.is_document_ready
    if not getattr(original, "_functionality_state_ready_guard", False):
        IngestionStateStore.is_document_ready = _wrap_document_ready(original)


__all__ = ["install"]
