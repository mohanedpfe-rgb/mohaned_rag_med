from __future__ import annotations

_INSTALLED = False
_TERMINAL = {"READY", "COMPLETED", "FAILED", "FAILED_EXTRACTION", "FAILED_OCR", "FAILED_EMBEDDING", "FAILED_INDEXING", "QUARANTINED", "DEGRADED_LEXICAL"}


def _transition_guard(self, document_id, new_stage, **values):
    record = self.get_document(document_id)
    if not record:
        raise ValueError(f"Document {document_id!r} does not exist.")
    current = str(record.get("current_stage") or record.get("status") or "").upper()
    target = str(new_stage or "").upper()
    if current in _TERMINAL and target not in _TERMINAL and target not in {"INTERRUPTED", "RECOVERING", "SUPERSEDED"}:
        raise RuntimeError(f"Terminal document cannot transition: {current} -> {target}.")
    if current in {"READY", "COMPLETED"} and target in {"INTERRUPTED", "RECOVERING"}:
        from rag_project.ingestion.state_store import utc_now
        with self._connect() as connection:
            connection.execute("UPDATE documents SET current_stage=?, status=?, index_state='FAILED', modified_at=? WHERE document_id=?", (target, target, utc_now(), str(document_id)))
        return
    return self._runtime_v5_original_transition(document_id, new_stage, **values)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.ingestion.state_store import IngestionStateStore
    if not hasattr(IngestionStateStore, "_runtime_v5_original_transition"):
        IngestionStateStore._runtime_v5_original_transition = IngestionStateStore.transition_document_state
    IngestionStateStore.transition_document_state = _transition_guard
    _INSTALLED = True
