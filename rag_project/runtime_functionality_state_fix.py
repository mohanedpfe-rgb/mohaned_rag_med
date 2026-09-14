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


def _install_terminal_recovery_guard() -> None:
    from rag_project.ingestion.state_store import IngestionStateStore
    original = IngestionStateStore.transition_document_state
    if getattr(original, "_functionality_terminal_recovery_guard", False):
        return

    def transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
        record = self.get_document(document_id)
        if not record:
            raise ValueError(f"Document {document_id!r} does not exist.")
        current = str(record.get("current_stage") or record.get("status") or "").upper()
        target = str(new_stage or "").upper()
        if current in {"READY", "COMPLETED"} and target in {"INTERRUPTED", "RECOVERING"}:
            from rag_project.ingestion.state_store import utc_now
            with self._connect() as connection:
                connection.execute("UPDATE documents SET current_stage=?, status=?, index_state='FAILED', modified_at=? WHERE document_id=?", (target, target, utc_now(), str(document_id)))
            return
        return original(self, document_id, new_stage, **values)

    transition._functionality_terminal_recovery_guard = True
    IngestionStateStore.transition_document_state = transition


def _install_page_reader() -> None:
    """Provide the page-checkpoint read side of the durable state contract."""
    from rag_project.ingestion.state_store import IngestionStateStore
    if callable(getattr(IngestionStateStore, "get_pages", None)):
        return
    def get_pages(self: Any, document_id: str) -> list[dict[str, Any]]:
        if not document_id:
            return []
        with self._connect() as connection:
            rows = connection.execute("SELECT document_id, page_number, extraction_status, ocr_status, extraction_method, text, cache_reference, processing_error, checksum, updated_at FROM pages WHERE document_id = ? ORDER BY page_number ASC", (str(document_id),)).fetchall()
        return [dict(row) for row in rows]
    IngestionStateStore.get_pages = get_pages


_install_terminal_recovery_guard()


def install() -> None:
    from rag_project.ingestion.state_store import IngestionStateStore
    original = IngestionStateStore.is_document_ready
    if not getattr(original, "_functionality_state_ready_guard", False):
        IngestionStateStore.is_document_ready = _wrap_document_ready(original)
    _install_page_reader()
    _install_terminal_recovery_guard()


__all__ = ["install"]
