from __future__ import annotations

import threading
from typing import Any


def install() -> None:
    """Restore the cancel-flag lifecycle expected by the canonical ingestor."""
    from rag_project.app.rag_system import (
        RAGSystem,
        _INGEST_CANCEL_FLAGS,
        _INGEST_LOCK,
        _IngestCancelFlag,
    )

    if getattr(RAGSystem, "_cancel_flag_contract_v1", False):
        return

    def _new_cancel_flag(self: Any, document_id: str) -> _IngestCancelFlag:
        key = str(document_id)
        with _INGEST_LOCK:
            existing = _INGEST_CANCEL_FLAGS.get(key)
            if existing is None or existing.cancelled:
                existing = _IngestCancelFlag()
                _INGEST_CANCEL_FLAGS[key] = existing
            return existing

    def _remove_cancel_flag(self: Any, document_id: str) -> None:
        with _INGEST_LOCK:
            _INGEST_CANCEL_FLAGS.pop(str(document_id), None)

    def cancel_ingest(self: Any, document_id: str) -> bool:
        with _INGEST_LOCK:
            flag = _INGEST_CANCEL_FLAGS.get(str(document_id))
        if flag is None:
            return False
        flag.cancel()
        return True

    RAGSystem._new_cancel_flag = _new_cancel_flag
    RAGSystem._remove_cancel_flag = _remove_cancel_flag
    RAGSystem.cancel_ingest = cancel_ingest
    RAGSystem._cancel_flag_contract_v1 = True


__all__ = ["install"]
