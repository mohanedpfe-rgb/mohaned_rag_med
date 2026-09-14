from __future__ import annotations

import os
import shutil
import unicodedata
from pathlib import Path
from typing import Any


def _lease_is_valid(record: dict[str, Any] | None) -> bool:
    if not record or not record.get("lease_owner") or not record.get("lease_expires_at"):
        return False
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat(str(record["lease_expires_at"]).replace("Z", "+00:00")) > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def _sanitize_evidence(text: str) -> str:
    if not text:
        return text
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    control = {ord(char): None for char in normalized if ord(char) < 32 and char not in "\n\t\r"}
    normalized = normalized.translate(control)
    suspicious = (
        "ignore previous instructions", "ignore all instructions", "disregard previous instructions",
        "override system prompt", "reveal system prompt", "developer mode", "admin override", "jailbreak",
    )
    lines: list[str] = []
    for line in normalized.splitlines():
        folded = " ".join(line.casefold().split())
        role_prefix = folded.startswith(("assistant:", "system:", "developer:", "instruction:", "user:"))
        lines.append("[REDACTED: source instruction-like text]" if role_prefix or any(marker in folded for marker in suspicious) else line)
    return "\n".join(lines).strip()


def _quarantine_processed_failure(system: Any, pdf_path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    if str(result.get("status", "")).lower() != "failed":
        return result
    source = Path(pdf_path)
    processed = Path(system.settings.processed_dir) / source.name
    failed = Path(system.settings.failed_dir) / source.name
    failed.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (source, processed):
        if not candidate.exists() or candidate.resolve() == failed.resolve():
            continue
        tmp = failed.with_name(f".{failed.name}.{os.getpid()}.part")
        try:
            if candidate.stat().st_dev == failed.parent.stat().st_dev:
                os.replace(candidate, tmp)
            else:
                shutil.copy2(candidate, tmp)
                candidate.unlink()
            os.replace(tmp, failed)
            break
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return result


def install() -> None:
    """Install data-safety helpers only.

    Lease validation is intentionally owned by the ingestion transaction/orchestrator,
    not by IngestionStateStore.transition_document_state. This module therefore does
    not monkey-patch state-store, vector-store, RAGSystem, or hardening methods.
    """
    from rag_project.app import rag_system as rag_module
    rag_module.sanitize_evidence = _sanitize_evidence


__all__ = ["install", "_lease_is_valid", "_sanitize_evidence", "_quarantine_processed_failure"]
