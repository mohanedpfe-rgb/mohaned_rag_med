from __future__ import annotations

_INSTALLED = False
_TERMINAL = {"READY", "COMPLETED", "FAILED", "FAILED_EXTRACTION", "FAILED_OCR", "FAILED_EMBEDDING", "FAILED_INDEXING", "QUARANTINED", "DEGRADED_LEXICAL"}


def _guard(self, document_id, new_stage, **values):
    record = self.get_document(document_id)
    if not record:
        raise ValueError(f"Document {document_id!r} does not exist.")
    current = str(record.get("current_stage") or record.get("status") or "").upper()
    target = str(new_stage or "").upper()
    if current in _TERMINAL and target not in _TERMINAL and target not in {"INTERRUPTED", "RECOVERING"}:
        raise RuntimeError(f"Invalid terminal state regression: {current} -> {target}.")
    if current in {"READY", "COMPLETED"} and target in {"INTERRUPTED", "RECOVERING"}:
        values = dict(values)
        values.update({"status": target, "current_stage": target, "index_state": "FAILED"})
        type(self)._runtime_v5_original_update_document(self, document_id, **values)
        return
    return self._runtime_v5_original_transition(document_id, new_stage, **values)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.ingestion.state_store import IngestionStateStore
    if not hasattr(IngestionStateStore, "_runtime_v5_original_transition"):
        IngestionStateStore._runtime_v5_original_transition = IngestionStateStore.transition_document_state
    if not hasattr(IngestionStateStore, "_runtime_v5_original_update_document"):
        IngestionStateStore._runtime_v5_original_update_document = IngestionStateStore.update_document
    IngestionStateStore.transition_document_state = _guard
    _INSTALLED = True
