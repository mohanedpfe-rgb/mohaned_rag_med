from __future__ import annotations

_INSTALLED = False

def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from rag_project.ingestion.state_store import IngestionStateStore
    if not hasattr(IngestionStateStore, "_runtime_v5_original_transition"):
        IngestionStateStore._runtime_v5_original_transition = IngestionStateStore.transition_document_state
    _INSTALLED = True
