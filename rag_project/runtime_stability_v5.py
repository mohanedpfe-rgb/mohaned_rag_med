from __future__ import annotations

import threading

_LOCK = threading.RLock()
_INSTALLED = False

_TERMINAL = {
    "READY", "COMPLETED", "FAILED", "FAILED_EXTRACTION", "FAILED_OCR",
    "FAILED_EMBEDDING", "FAILED_INDEXING", "QUARANTINED", "DEGRADED_LEXICAL",
}


def _transition_guard(self, document_id, new_stage, **values):
    record = self.get_document(document_id)
    if not record:
        raise ValueError(f"Document {document_id!r} does not exist.")
    current = str(record.get("current_stage") or record.get("status") or "").upper()
    target = str(new_stage or "").upper()
    # Terminal records are immutable through transition_document_state. A new
    # ingestion must first atomically claim/update the document to a non-terminal
    # state; this prevents stale/foreign leases from bypassing the state fence.
    if current in _TERMINAL and target not in _TERMINAL:
        raise RuntimeError(f"Invalid terminal state regression: {current} -> {target}.")
    return self._runtime_v5_original_transition(document_id, new_stage, **values)


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.ingestion.state_store import IngestionStateStore
        if not hasattr(IngestionStateStore, "_runtime_v5_original_transition"):
            IngestionStateStore._runtime_v5_original_transition = IngestionStateStore.transition_document_state
            IngestionStateStore.transition_document_state = _transition_guard
        _INSTALLED = True
