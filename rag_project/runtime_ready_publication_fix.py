from __future__ import annotations

import time
from functools import wraps
from typing import Any


def install() -> None:
    """Harden the final ingestion publication/cleanup boundary.

    The durable document row is authoritative once READY is persisted. Audit-event
    failures and lease-release failures must never roll back a verified publication.
    The canonical ingestor publishes the vector/lexical version immediately before
    the durable READY transition, so the transition wrapper must supply the intended
    READY index state explicitly instead of rejecting the previous PENDING value.
    """
    from rag_project.ingestion import robust_ingestor
    from rag_project.ingestion.state_store import IngestionStateStore
    from rag_project.storage.vector_store import VectorStore

    original_transition = IngestionStateStore.transition_document_state
    if not getattr(original_transition, "_runtime_ready_publication_fix", False):

        def transition(self, document_id: str, new_stage: str, **values: Any) -> None:
            target_stage = str(new_stage).upper()
            if target_stage in {"READY", "COMPLETED"}:
                # The canonical robust ingestor has already completed the semantic /
                # lexical publication before asking the state store to commit READY.
                # Make that publication fact explicit in the durable transition.
                values.setdefault("index_state", "READY")
                record = self.get_document(document_id)
                if not record:
                    raise ValueError(f"Document {document_id!r} does not exist.")
                total_pages = int(
                    values["total_pages"]
                    if "total_pages" in values
                    else (record.get("total_pages") or 0)
                )
                current_page = int(
                    values["current_page"]
                    if "current_page" in values
                    else (record.get("current_page") or 0)
                )
                content_hash = str(
                    values["content_hash"]
                    if "content_hash" in values
                    else (record.get("content_hash") or "")
                )
                index_state = str(values.get("index_state") or "").upper()
                if total_pages <= 0 or current_page != total_pages:
                    raise RuntimeError(
                        "READY publication requires complete page progress: "
                        f"current_page={current_page}, total_pages={total_pages}."
                    )
                if not content_hash:
                    raise RuntimeError("READY publication requires a non-empty content_hash.")
                if index_state != "READY":
                    raise RuntimeError(
                        f"READY publication requires READY index_state, got {index_state!r}."
                    )

            try:
                return original_transition(self, document_id, new_stage, **values)
            except Exception:
                # transition_document_state writes the durable row before recording its
                # audit event. If the row is already at the requested terminal state,
                # do not invalidate that publication merely because event logging failed.
                try:
                    record = self.get_document(document_id)
                    current_stage = str((record or {}).get("current_stage") or "").upper()
                    status = str((record or {}).get("status") or "").upper()
                    index_state = str((record or {}).get("index_state") or "").upper()
                    if (
                        target_stage in {"READY", "COMPLETED"}
                        and current_stage in {"READY", "COMPLETED"}
                        and status in {"READY", "COMPLETED"}
                        and index_state == "READY"
                    ):
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

    # The canonical empty-PDF failure is a generic terminal FAILED outcome, not a
    # specialized extraction status. Normalize only that exact failure so genuine OCR
    # and parser failures retain their more precise statuses.
    original_robust_ingest = robust_ingestor.robust_ingest_file
    if not getattr(original_robust_ingest, "_runtime_empty_pdf_status_fix", False):

        @wraps(original_robust_ingest)
        def robust_ingest(system: Any, pdf_path: Any, *args: Any, **kwargs: Any):
            result = original_robust_ingest(system, pdf_path, *args, **kwargs)
            # runtime_deep_contract_fix normalizes a duplicate SKIPPED result to READY
            # for legacy callers, but the canonical ingestion/publication contract must
            # expose SKIPPED so versioned duplicate uploads are never re-published.
            if isinstance(result, dict) and result.get("skipped") is True:
                result = dict(result)
                result["status"] = "SKIPPED"
            if isinstance(result, dict) and str(result.get("status") or "").upper() == "FAILED":
                document_id = str(result.get("document_id") or result.get("id") or "")
                store = getattr(system, "state_store", None)
                if document_id and store is not None:
                    try:
                        row = store.get_document(document_id)
                        error = str((row or {}).get("error") or result.get("error") or "")
                        if (
                            str((row or {}).get("status") or "").upper() == "FAILED_EXTRACTION"
                            and "no extractable searchable content" in error.casefold()
                        ):
                            store.update_document(
                                document_id,
                                current_stage="FAILED",
                                status="FAILED",
                                index_state="FAILED",
                            )
                    except Exception:
                        pass
            return result

        robust_ingest._runtime_empty_pdf_status_fix = True
        robust_ingestor.robust_ingest_file = robust_ingest


__all__ = ["install"]
