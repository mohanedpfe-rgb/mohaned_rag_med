from __future__ import annotations

_INSTALLED = False
_TERMINAL = {"READY", "COMPLETED", "FAILED", "FAILED_EXTRACTION", "FAILED_OCR", "FAILED_EMBEDDING", "FAILED_INDEXING", "QUARANTINED", "DEGRADED_LEXICAL"}


def _transition_guard(self, document_id, new_stage, **values):
    record = self.get_document(document_id)
    if not record:
        raise ValueError(f"Document {document_id!r} does not exist.")
    current = str(record.get("current_stage") or record.get("status") or "").upper()
    target = str(new_stage or "").upper()
    if current in _TERMINAL and target not in _TERMINAL and target not in {"INTERRUPTED", "RECOVERING"}:
        raise RuntimeError(f"Terminal document cannot transition: {current} -> {target}.")
    if current in {"READY", "COMPLETED"} and target in {"INTERRUPTED", "RECOVERING"}:
        with self._connect() as connection:
            from rag_project.ingestion.state_store import utc_now
            connection.execute("UPDATE documents SET current_stage=?, status=?, index_state='FAILED', modified_at=? WHERE document_id=?", (target, target, utc_now(), str(document_id)))
        return
    return self._runtime_v5_original_transition(document_id, new_stage, **values)


def _release_guard(self, document_id, worker_id):
    try:
        return bool(self._runtime_v5_original_release_document(document_id, worker_id))
    except Exception:
        # Lease release is cleanup. It must not replace the primary ingestion
        # outcome or escape the final cleanup boundary.
        return False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.ingestion.state_store import IngestionStateStore
    if not hasattr(IngestionStateStore, "_runtime_v5_original_transition"):
        IngestionStateStore._runtime_v5_original_transition = IngestionStateStore.transition_document_state
    if not hasattr(IngestionStateStore, "_runtime_v5_original_release_document"):
        IngestionStateStore._runtime_v5_original_release_document = IngestionStateStore.release_document
    IngestionStateStore.transition_document_state = _transition_guard
    IngestionStateStore.release_document = _release_guard
    _INSTALLED = True


__all__ = ["install"]