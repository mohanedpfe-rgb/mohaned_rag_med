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

    original_upsert = IngestionStateStore.upsert_document
    if not getattr(original_upsert, "_terminal_ready_upsert_guard", False):

        @wraps(original_upsert)
        def upsert(self, values: dict[str, Any]) -> None:
            requested = dict(values or {})
            document_id = requested.get("document_id")
            current = self.get_document(str(document_id)) if document_id else None
            if current and _normalized(current.get("status")) in {"READY", "COMPLETED"}:
                current_hash = str(current.get("content_hash") or "")
                current_version = str(current.get("version_id") or "")
                requested_hash = str(requested.get("content_hash") or "")
                requested_version = str(requested.get("version_id") or "")
                requested_status = _normalized(requested.get("status"))
                same_published_version = bool(
                    requested_hash
                    and requested_hash == current_hash
                    and requested_version
                    and requested_version == current_version
                )
                # A READY row is immutable only for the exact published
                # version. A new content hash/version is a legitimate
                # replacement flow and must be allowed to enter RUNNING so
                # transactional ingestion can build and publish it (or roll
                # back to the older READY version if publication fails).
                if same_published_version and requested_status in _FAILURE_STATUSES:
                    raise RuntimeError(
                        f"READY document {document_id!r} cannot be overwritten by "
                        f"status={requested_status or 'UNSPECIFIED'} for the same published version."
                    )
                if same_published_version and requested_status not in {"READY", "COMPLETED", "SUPERSEDED"}:
                    raise RuntimeError(
                        f"READY document {document_id!r} cannot be overwritten by "
                        f"status={requested_status or 'UNSPECIFIED'} for the same published version."
                    )
            return original_upsert(self, requested)

        upsert._terminal_ready_upsert_guard = True
        IngestionStateStore.upsert_document = upsert

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
            if current_status in {"READY", "COMPLETED"} and target == "SUPERSEDED":
                values = dict(values)
                values.setdefault("current_stage", "SUPERSEDED")
                values.setdefault("status", "SUPERSEDED")
                values.setdefault("index_state", "FAILED")
                original_update(self, document_id, **values)
                try:
                    self.record_event(
                        document_id,
                        stage="SUPERSEDED",
                        status="SUPERSEDED",
                        event_type="version_retired",
                        message="Document version explicitly superseded.",
                        details={"transition": "READY->SUPERSEDED"},
                    )
                except Exception:
                    pass
                return None
            return original_transition(self, document_id, new_stage, **values)

        transition._terminal_ready_transition_guard = True
        IngestionStateStore.transition_document_state = transition


__all__ = ["install"]