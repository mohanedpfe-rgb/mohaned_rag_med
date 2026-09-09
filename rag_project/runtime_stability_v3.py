from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()
_INSTALLED = False
_PROCESS_INGEST_LOCK = threading.RLock()
_ANSWER_LOCK = threading.RLock()


def _stable_production_ingest(self: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Route the production subclass through the final ingestion guard."""
    from rag_project.runtime_stability_v2 import _safe_ingest_file

    with _PROCESS_INGEST_LOCK:
        return _safe_ingest_file(self, pdf_path)


def _health_report_fast(self: Any) -> dict[str, Any]:
    """Non-blocking health snapshot; never load a model or make an embedding request."""
    embedding_service = self.embedding_service
    available = getattr(embedding_service, "_ollama_available", None)
    last_error = getattr(embedding_service, "last_error", None)
    try:
        identity = embedding_service.identity
    except Exception:
        identity = None
    try:
        index = self.vector_store.compatibility_report(identity) if identity is not None else {
            "status": "UNKNOWN",
            "message": "Embedding identity has not been discovered yet.",
        }
    except Exception as exc:
        index = {"status": "UNAVAILABLE", "error": str(exc)}
    try:
        vector_count = int(self.vector_store.count())
    except Exception as exc:
        vector_count = None
        audit = {"ok": False, "error": str(exc), "mode": "lightweight"}
    else:
        audit = {"ok": True, "mode": "lightweight", "vector_count": vector_count}
    return {
        "ready": bool(available is True and index.get("status") in {"READY", "OK"}),
        "embedding": {
            "ok": available is True,
            "identity": identity,
            "error": None if available is not False else last_error,
            "dimension": getattr(embedding_service, "dimension", None),
        },
        "index": index,
        "audit": audit,
        "feature_contract": getattr(self, "_production_feature_contract", {"all_resolved": True}),
        "models": {"embedding_model": self.settings.embedding_model, "generation_model": self.settings.generation_model},
        "pipeline": {"explicit_composition": True, "non_blocking_health": True},
    }


def _locked_answer(self: Any, question: str, metadata_filter=None):
    """Serialize expensive local generation to avoid concurrent Ollama overload."""
    with _ANSWER_LOCK:
        return self._runtime_v3_original_answer(question, metadata_filter)


def _guard_transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
    """Reject impossible stage regressions and stale page-counter rollbacks."""
    record = self.get_document(document_id)
    if not record:
        raise ValueError(f"Document {document_id!r} does not exist.")
    current = str(record.get("current_stage") or "DISCOVERED").upper()
    target = str(new_stage).upper()
    terminal = {
        "READY", "COMPLETED", "FAILED", "FAILED_EXTRACTION", "FAILED_OCR",
        "FAILED_EMBEDDING", "FAILED_INDEXING", "DEGRADED_LEXICAL", "QUARANTINED",
    }
    if current in {"READY", "COMPLETED"} and target not in terminal:
        raise RuntimeError(f"Invalid state regression: {current} -> {target}")
    if current.startswith("FAILED") and target not in terminal:
        raise RuntimeError(f"Invalid state regression: {current} -> {target}")
    current_page = int(record.get("current_page") or 0)
    if "current_page" in values:
        requested = int(values["current_page"] or 0)
        if requested < current_page and target not in terminal | {"INTERRUPTED", "RECOVERING"}:
            values["current_page"] = current_page
    return self._runtime_v3_original_transition(document_id, target, **values)


def _merge_metrics(document: dict[str, Any]) -> dict[str, Any]:
    """Promote durable metrics into document fields expected by the UI."""
    raw = document.get("ingestion_metrics")
    metrics: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                metrics = parsed
        except (TypeError, ValueError):
            metrics = {}
    for field in ("chunk_count", "embedding_count", "page_count", "total", "elapsed_ms", "total_ms"):
        if document.get(field) in (None, "") and field in metrics:
            document[field] = metrics[field]
    if "chunk_count" not in document and "chunks" in metrics:
        document["chunk_count"] = metrics["chunks"]
    if "embedding_count" not in document and "embeddings" in metrics:
        document["embedding_count"] = metrics["embeddings"]
    return document


def _enriched_documents(self: Any) -> list[dict[str, Any]]:
    return [_merge_metrics(dict(row)) for row in self._runtime_v3_original_get_all_documents()]


def _enriched_document(self: Any, document_id: str) -> dict[str, Any] | None:
    row = self._runtime_v3_original_get_document(document_id)
    return _merge_metrics(dict(row)) if row else None


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return

        from rag_project.app.production_rag import ProductionRAGSystem
        from rag_project.ingestion.state_store import IngestionStateStore

        if not hasattr(ProductionRAGSystem, "_runtime_v3_original_ingest_file"):
            ProductionRAGSystem._runtime_v3_original_ingest_file = ProductionRAGSystem.ingest_file
            ProductionRAGSystem.ingest_file = _stable_production_ingest

        if not hasattr(ProductionRAGSystem, "_runtime_v3_original_answer"):
            ProductionRAGSystem._runtime_v3_original_answer = ProductionRAGSystem.answer
            ProductionRAGSystem.answer = _locked_answer

        if not hasattr(ProductionRAGSystem, "_runtime_v3_detailed_health_report"):
            ProductionRAGSystem._runtime_v3_detailed_health_report = ProductionRAGSystem.health_report
            ProductionRAGSystem.health_report_fast = _health_report_fast
            ProductionRAGSystem.health_report = _health_report_fast

        if not hasattr(IngestionStateStore, "_runtime_v3_original_transition"):
            IngestionStateStore._runtime_v3_original_transition = IngestionStateStore.transition_document_state
            IngestionStateStore.transition_document_state = _guard_transition

        if not hasattr(IngestionStateStore, "_runtime_v3_original_get_all_documents"):
            IngestionStateStore._runtime_v3_original_get_all_documents = IngestionStateStore.get_all_documents
            IngestionStateStore.get_all_documents = _enriched_documents

        if not hasattr(IngestionStateStore, "_runtime_v3_original_get_document"):
            IngestionStateStore._runtime_v3_original_get_document = IngestionStateStore.get_document
            IngestionStateStore.get_document = _enriched_document

        _INSTALLED = True
