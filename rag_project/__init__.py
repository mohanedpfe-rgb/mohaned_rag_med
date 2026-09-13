"""Production-grade Python RAG application package."""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"


# Compatibility repair for durable production boundaries.
# Keep the canonical implementation strict while ensuring legacy callers
# bind the required primary-key fields explicitly.
from .ingestion import state_store as _state_store


def _fixed_upsert_page(self, document_id: str, page_number: int, **values):
    """Create or update one page checkpoint with both primary-key fields bound."""
    if not document_id:
        raise ValueError("document_id must be non-empty")
    try:
        page_number = int(page_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("page_number must be an integer") from exc
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    unknown = set(values) - _state_store._ALLOWED_PAGE_UPDATE_KEYS
    if unknown:
        raise ValueError(f"Unsupported page update field(s): {', '.join(sorted(unknown))}")
    if self.get_document(document_id) is None:
        raise ValueError(f"Document {document_id!r} does not exist.")

    values = dict(values)
    values.setdefault("extraction_status", "PENDING")
    values.setdefault("ocr_status", "PENDING")
    values.setdefault("updated_at", _state_store.utc_now())

    selected = {
        "document_id": document_id,
        "page_number": page_number,
        **{key: values[key] for key in _state_store._ALLOWED_PAGE_UPDATE_KEYS if key in values},
    }
    placeholders = ", ".join("?" for _ in selected)
    assignments = ", ".join(
        f"{key}=excluded.{key}"
        for key in selected
        if key not in {"document_id", "page_number"}
    )
    with self._connect() as connection:
        connection.execute(
            f"INSERT INTO pages ({', '.join(selected)}) VALUES ({placeholders}) "
            f"ON CONFLICT(document_id, page_number) DO UPDATE SET {assignments}",
            tuple(selected.values()),
        )


_original_transition_document_state = _state_store.IngestionStateStore.transition_document_state


def _fixed_transition_document_state(self, document_id: str, new_stage: str, **values):
    """Keep the READY invariant while making the documented READY default reachable."""
    if str(new_stage).upper() in {"READY", "COMPLETED"} and "index_state" not in values:
        values["index_state"] = "READY"
    return _original_transition_document_state(self, document_id, new_stage, **values)


_state_store.IngestionStateStore.upsert_page = _fixed_upsert_page
_state_store.IngestionStateStore.transition_document_state = _fixed_transition_document_state
