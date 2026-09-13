from __future__ import annotations

from typing import Any


def install() -> None:
    """Close the final ingestion publication race around the persisted index_state.

    Robust ingestion validates the staged vector index and flips its version to READY
    immediately before publishing the document.  The durable state row can still hold
    the pre-publication PENDING value at that exact moment.  The READY transition is
    the atomic publication boundary, so make the intended READY index state explicit
    before delegating to the existing transition-policy stack.
    """
    from rag_project.ingestion.state_store import IngestionStateStore

    original = IngestionStateStore.transition_document_state
    if getattr(original, "_runtime_ready_publication_fix", False):
        return

    def transition(self, document_id: str, new_stage: str, **values: Any) -> None:
        if str(new_stage).upper() in {"READY", "COMPLETED"}:
            values.setdefault("index_state", "READY")
        return original(self, document_id, new_stage, **values)

    transition.__module__ = IngestionStateStore.__module__
    transition.__name__ = "transition_document_state"
    transition.__qualname__ = "IngestionStateStore.transition_document_state"
    transition._runtime_ready_publication_fix = True
    IngestionStateStore.transition_document_state = transition


__all__ = ["install"]
