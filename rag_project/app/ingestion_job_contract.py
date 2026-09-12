from __future__ import annotations

from collections.abc import Iterable
from typing import Any


_SUCCESS_STATUSES = frozenset({"success", "completed", "ready", "skipped"})
_FAILURE_STATUSES = frozenset({"failed", "failed_embedding", "failed_indexing", "error"})


def summarize_ingestion_results(results: Iterable[dict[str, Any]]) -> tuple[int, int, str]:
    """Normalize ingestion result statuses at the UI job boundary.

    The production service may expose READY/COMPLETED while older adapters expose
    success/skipped. The UI job state must count both forms consistently.
    Unknown statuses are not silently classified as failures because that would
    turn a newly introduced non-terminal state into a false error.
    """
    completed = 0
    failed = 0
    for item in results:
        status = str(item.get("status") or "").strip().casefold()
        if status in _SUCCESS_STATUSES:
            completed += 1
        elif status in _FAILURE_STATUSES:
            failed += 1

    if failed and not completed:
        job_status = "FAILED"
    elif failed:
        job_status = "COMPLETED_WITH_FAILURES"
    else:
        job_status = "COMPLETED"
    return completed, failed, job_status
