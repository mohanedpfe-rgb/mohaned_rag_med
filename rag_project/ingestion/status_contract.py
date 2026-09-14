from __future__ import annotations

from typing import Any


PUBLIC_READY_STATUSES = frozenset({"SUCCESS", "COMPLETED", "READY"})
PUBLIC_FAILURE_STATUSES = frozenset({
    "FAILED",
    "FAILED_EXTRACTION",
    "FAILED_OCR",
    "FAILED_EMBEDDING",
    "FAILED_INDEXING",
    "QUARANTINED",
})


def normalize_public_status(value: Any) -> str:
    """Map internal ingestion outcomes to the single public status vocabulary."""
    status = str(value or "").strip().upper()
    if status in PUBLIC_READY_STATUSES:
        return "READY"
    if status in PUBLIC_FAILURE_STATUSES:
        return "FAILED"
    if status == "SUPERSEDED":
        return "SUPERSEDED"
    if status in {"SKIPPED", "DUPLICATE"}:
        return "SKIPPED"
    return status or "UNKNOWN"


def public_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy with exactly one public ingestion status mapping."""
    out = dict(result or {})
    out["status"] = normalize_public_status(out.get("status"))
    return out


def is_public_success(result: dict[str, Any] | None) -> bool:
    return normalize_public_status((result or {}).get("status")) == "READY"


__all__ = ["PUBLIC_READY_STATUSES", "PUBLIC_FAILURE_STATUSES", "normalize_public_status", "public_result", "is_public_success"]
