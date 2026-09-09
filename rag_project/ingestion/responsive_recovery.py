from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTIVE_STATUSES = {
    "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR",
    "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING",
    "INTERRUPTED", "RECOVERING",
}


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def recover_orphaned_documents(system: Any, working_paths: set[str], *, stale_seconds: float = 90.0) -> int:
    """Recover active DB jobs left behind by a dead/reloaded application process.

    A job currently owned by this process is never touched. Jobs owned by a missing
    in-process worker and lacking a fresh heartbeat are fenced into INTERRUPTED so
    the supervisor can safely retry the PDF instead of waiting for the full lease.
    """
    now = datetime.now(timezone.utc)
    recovered = 0
    try:
        documents = list(system.state_store.get_all_documents() or [])
    except Exception:
        return 0

    for document in documents:
        status = str(document.get("status") or "").upper()
        if status not in ACTIVE_STATUSES:
            continue
        path_value = document.get("file_path")
        if not path_value:
            continue
        try:
            path_key = str(Path(path_value).resolve())
        except (OSError, ValueError):
            continue
        if path_key in working_paths:
            continue

        heartbeat = _parse(document.get("heartbeat_at")) or _parse(document.get("modified_at"))
        if heartbeat is None:
            age = stale_seconds
        else:
            age = max(0.0, (now - heartbeat).total_seconds())
        if age < stale_seconds:
            continue

        try:
            system.state_store.update_document(
                str(document.get("document_id")),
                current_stage="INTERRUPTED",
                status="INTERRUPTED",
                index_state="FAILED",
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error=f"Recovered orphaned ingestion after {int(age)}s without a heartbeat.",
            )
            recovered += 1
        except Exception:
            continue
    return recovered


__all__ = ["recover_orphaned_documents"]
