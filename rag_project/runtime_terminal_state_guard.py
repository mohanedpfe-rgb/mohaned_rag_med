from __future__ import annotations

from functools import wraps
from typing import Any


_FAILURE_STATUSES = {
    "FAILED",
    "FAILED_EXTRACTION",
    "FAILED_OCR",
    "FAILED_EMBEDDING",
    "FAILED_INDEXING",
    "DEGRADED_LEXICAL",
    "QUARANTINED",
    "RUNNING",
    "DISCOVERED",
    "VALIDATING",
    "EXTRACTING",
    "OCR",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "VALIDATING_INDEX",
    "INTERRUPTED",
    "RECOVERING",
}


def _normalized(value: Any) -> str:
    return str(value or "").upper()


def install() -> None:
    """Make durable READY publication monotonic except for explicit supersession."""
    from rag_project.ingestion.state_store import IngestionStateStore

    original_update = IngestionStateStore.update_document
    if not getattr(original_update, "_terminal_ready_guard", False):

        @wraps(original_update)
        def update(self, document_id: str, **values: Any) -> None:
            current = self.get_document(document_id)
            current_status = _normalized((current or {}).get("status"))
            requested_status = _normalized(values.get("status"))
            requested_stage = _normalized(values.get("current_stage"))
            if current_status in {"READY", "COMPLETED"} and (
                requested_status in _FAILURE_STATUSES
                or requested_stage in _FAILURE_STATUSES
            ):
                raise RuntimeError(
                    f"READY document {document_id!r} cannot regress to "
                    f"status={requested_status or current_status}, "
                    f"stage={requested_stage or (current or {}).get('current_stage')}"
                )
            return original_update(self, document_id, **values)

        update._terminal_ready_guard = True
        IngestionStateStore.update_document = update

    original_transition = IngestionStateStore.transition_document_state
    if not getattr(original_transition, "_terminal_ready_transition_guard", False):

        @wraps(original_transition)
        def transition(self, document_id: str, new_stage: str, **values: Any) -> None:
            current = self.get_document(document_id)
            current_status = _normalized((current or {}).get("status"))
            target = _normalized(new_stage)
            if current_status in {"READY", "COMPLETED"} and target in _FAILURE_STATUSES:
                raise RuntimeError(
                    f"READY document {document_id!r} cannot transition to {target}."
                )
            return original_transition(self, document_id, new_stage, **values)

        transition._terminal_ready_transition_guard = True
        IngestionStateStore.transition_document_state = transition


__all__ = ["install"]
