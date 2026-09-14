from __future__ import annotations

from typing import Any

_FAILURE_STATUSES = {
    "FAILED", "FAILED_EXTRACTION", "FAILED_OCR", "FAILED_EMBEDDING", "FAILED_INDEXING",
    "DEGRADED_LEXICAL", "QUARANTINED", "RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING",
    "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX",
}
_RECOVERY_TARGETS = {"INTERRUPTED", "RECOVERING"}


def _normalized(value: Any) -> str:
    return str(value or "").upper()


def install() -> None:
    """Compatibility marker only; terminal-state rules live in IngestionStateStore."""
    return None


__all__ = ["install"]
