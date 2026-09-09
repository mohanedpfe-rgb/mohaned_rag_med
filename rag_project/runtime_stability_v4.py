from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False
_THREAD = threading.local()
_QUERY_LOCK = threading.RLock()
_CLEAR_LOCK = threading.RLock()
_ACTIVE = ("RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "INTERRUPTED", "RECOVERING")


def _now():
    return datetime.now(timezone.utc)


def _workers():
    value = getattr(_THREAD, "workers", None)
    if value is None:
        value = {}
        _THREAD.workers = value
    return value


def _claim(self, document_id, worker_id, lease_seconds=900):
    ok = bool(self._runtime_v4_original_claim(document_id, worker_id, lease_seconds=lease_seconds))
    if ok:
        _workers()[str(document_id)] = str(worker_id)
    return ok


def _live(row):
    if not row or not row.get("lease_owner") or not row.get("lease_expires_at"):
        return False
    try:
        stamp = datetime.fromisoformat(str(row["lease_expires_at"]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp > _now()
    except (TypeError, ValueError):
        return False


def _fence(self, document_id):
    row = self.get_document(document_id)
    if not row:
        raise ValueError(f"Document {document_id!r} does not exist.")
    owner = row.get("lease_owner")
    if owner and (_workers().get(str(document_id)) != str(owner) or not _live(row)):
        raise RuntimeError(f"Stale or expired ingestion writer rejected for {document_id!r}.")


def _update(self, document_id, **values):
    _fence(self, document_id)
    return self._runtime_v4_original_update_document(document_id, **values)


def _page(self, document_id, page_number, **values):
    _fence(self, document_id)
    return self._runtime_v4_original_upsert_page(document_id, page_number, **values)


def _event(self, document_id, **kwargs):
    _fence(self, document_id)
    return self._runtime_v4_original_record_event(document_id, **kwargs)


def _has_live(self):
    marks = ",".join("?" for _ in _ACTIVE)
    with sqlite3.connect(self.database_path) as db:
        row = db.execute(f"SELECT 1 FROM documents WHERE status IN ({marks}) AND lease_owner IS NOT NULL AND lease_expires_at > ? LIMIT 1", (*_ACTIVE, _now().isoformat())).fetchone()
    return row is not None


def _clear_state(self):
    with _CLEAR_LOCK:
        if _has_live(self):
            raise RuntimeError("Cannot clear PDF data while ingestion is running.")
        return self._runtime_v4_original_clear_all()


def _clear_production(self):
    with _CLEAR_LOCK:
        if _has_live(self.state_store):
            raise RuntimeError("Cannot clear PDF data while ingestion is running.")
        return self._runtime_v4_original_production_clear()


def _answer(self, question, metadata_filter=None):
    # Production answers are serialized because local Ollama generation is resource heavy.
    with _QUERY_LOCK:
        return self._runtime_v4_original_answer(question, metadata_filter)


def install():
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.app.production_rag import ProductionRAGSystem
        from rag_project.ingestion.state_store import IngestionStateStore
        if not hasattr(IngestionStateStore, "_runtime_v4_original_claim"):
            IngestionStateStore._runtime_v4_original_claim = IngestionStateStore.claim_document
            IngestionStateStore.claim_document = _claim
        if not hasattr(IngestionStateStore, "_runtime_v4_original_update_document"):
            IngestionStateStore._runtime_v4_original_update_document = IngestionStateStore.update_document
            IngestionStateStore.update_document = _update
        if not hasattr(IngestionStateStore, "_runtime_v4_original_upsert_page"):
            IngestionStateStore._runtime_v4_original_upsert_page = IngestionStateStore.upsert_page
            IngestionStateStore.upsert_page = _page
        if not hasattr(IngestionStateStore, "_runtime_v4_original_record_event"):
            IngestionStateStore._runtime_v4_original_record_event = IngestionStateStore.record_event
            IngestionStateStore.record_event = _event
        if not hasattr(IngestionStateStore, "_runtime_v4_original_clear_all"):
            IngestionStateStore._runtime_v4_original_clear_all = IngestionStateStore.clear_all
            IngestionStateStore.clear_all = _clear_state
        if not hasattr(ProductionRAGSystem, "_runtime_v4_original_production_clear"):
            ProductionRAGSystem._runtime_v4_original_production_clear = ProductionRAGSystem.clear_pdf_data
            ProductionRAGSystem.clear_pdf_data = _clear_production
        if not hasattr(ProductionRAGSystem, "_runtime_v4_original_answer"):
            ProductionRAGSystem._runtime_v4_original_answer = ProductionRAGSystem.answer
            ProductionRAGSystem.answer = _answer
        _INSTALLED = True
