from __future__ import annotations

import time
from typing import Any


def install() -> None:
    """Harden the final ingestion publication/cleanup boundary.

    The durable document row is authoritative once READY is persisted. Audit-event
    failures and lease-release failures must never roll back a verified publication.
    Transient Chroma/SQLite state-update failures receive bounded retries.
    """
    from rag_project.ingestion.state_store import IngestionStateStore
    from rag_project.storage.vector_store import VectorStore

    original_transition = IngestionStateStore.transition_document_state
    if not getattr(original_transition, "_runtime_ready_publication_fix", False):

        def transition(self, document_id: str, new_stage: str, **values: Any) -> None:
            target_stage = str(new_stage).upper()
            if target_stage in {"READY", "COMPLETED"}:
                values.setdefault("index_state", "READY")
            try:
                return original_transition(self, document_id, new_stage, **values)
            except Exception:
                # transition_document_state writes the document before recording its
                # audit event. If the durable row is already at the requested terminal
                # state, do not invalidate that publication merely because event logging
                # failed under SQLite contention or shutdown.
                try:
                    record = self.get_document(document_id)
                    current_stage = str((record or {}).get("current_stage") or "").upper()
                    status = str((record or {}).get("status") or "").upper()
                    index_state = str((record or {}).get("index_state") or "").upper()
                    if target_stage in {"READY", "COMPLETED"} and current_stage in {"READY", "COMPLETED"} and status in {"READY", "COMPLETED"} and index_state == "READY":
                        return None
                    if current_stage == target_stage:
                        return None
                except Exception:
                    pass
                raise

        transition.__module__ = IngestionStateStore.__module__
        transition.__name__ = "transition_document_state"
        transition.__qualname__ = "IngestionStateStore.transition_document_state"
        transition._runtime_ready_publication_fix = True
        IngestionStateStore.transition_document_state = transition

    original_release = IngestionStateStore.release_document
    if not getattr(original_release, "_runtime_release_cleanup_fix", False):

        def release(self, document_id: str, worker_id: str) -> bool:
            for attempt in range(3):
                try:
                    return bool(original_release(self, document_id, worker_id))
                except Exception:
                    if attempt < 2:
                        time.sleep(0.05 * (attempt + 1))
            # Lease cleanup is not allowed to convert a successful publication into
            # an ingestion failure. A stale lease is recoverable by the supervisor.
            return False

        release.__module__ = IngestionStateStore.__module__
        release.__name__ = "release_document"
        release.__qualname__ = "IngestionStateStore.release_document"
        release._runtime_release_cleanup_fix = True
        IngestionStateStore.release_document = release

    original_set_version = VectorStore.set_version_index_state
    if not getattr(original_set_version, "_runtime_version_state_retry_fix", False):

        def set_version_state(self, document_id: str, version_id: str, state: str) -> None:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    return original_set_version(self, document_id, version_id, state)
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(0.10 * (attempt + 1))
            if last_error is not None:
                raise last_error

        set_version_state.__module__ = VectorStore.__module__
        set_version_state.__name__ = "set_version_index_state"
        set_version_state.__qualname__ = "VectorStore.set_version_index_state"
        set_version_state._runtime_version_state_retry_fix = True
        VectorStore.set_version_index_state = set_version_state


__all__ = ["install"]
