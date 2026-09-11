from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()
_INSTALLED = False
_PROCESS_INGEST_LOCK = threading.RLock()
_ANSWER_LOCK = threading.RLock()


def _stable_production_ingest(self: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Compatibility bridge: production now has one canonical ingestion path."""
    from rag_project.ingestion.robust_ingestor import robust_ingest_file

    with _PROCESS_INGEST_LOCK:
        return robust_ingest_file(self, pdf_path)


def _safe_clear(self: Any) -> None:
    """Do not clear storage while a production ingestion owns the process lock."""
    if not _PROCESS_INGEST_LOCK.acquire(timeout=0.25):
        raise RuntimeError("Cannot clear PDF data while ingestion is running. Wait until processing finishes and try again.")
    try:
        return self._runtime_v3_original_clear()
    finally:
        _PROCESS_INGEST_LOCK.release()


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
        index = self.vector_store.compatibility_report(identity) if identity is not None else {"status": "UNKNOWN", "message": "Embedding identity has not been discovered yet."}
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
        "embedding": {"ok": available is True, "identity": identity, "error": None if available is not False else last_error, "dimension": getattr(embedding_service, "dimension", None)},
        "index": index,
        "audit": audit,
        "feature_contract": getattr(self, "_production_feature_contract", {"all_resolved": True}),
        "models": {"embedding_model": self.settings.embedding_model, "generation_model": self.settings.generation_model},
        "pipeline": {"explicit_composition": True, "non_blocking_health": True},
    }


def _locked_answer(self: Any, question: str, metadata_filter=None):
    with _ANSWER_LOCK:
        return self._runtime_v3_original_answer(question, metadata_filter)


def _guard_transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
    record = self.get_document(document_id)
    if not record:
        raise ValueError(f"Document {document_id!r} does not exist.")
    current = str(record.get("current_stage") or "DISCOVERED").upper()
    target = str(new_stage).upper()
    terminal = {"READY", "COMPLETED", "FAILED", "FAILED_EXTRACTION", "FAILED_OCR", "FAILED_EMBEDDING", "FAILED_INDEXING", "DEGRADED_LEXICAL", "QUARANTINED"}
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


def _bounded_discover_dimension(self: Any) -> int:
    original_timeout = float(self.timeout_seconds)
    original_retries = int(self.retries)
    try:
        if not self._check_ollama_available(force=False):
            raise RuntimeError(f"Embedding backend unavailable: Ollama at {self.base_url!r} did not respond to its health check.")
        self.timeout_seconds = min(original_timeout, float(os.getenv("RAG_DIMENSION_TIMEOUT", "15")))
        self.retries = 0
        return self._runtime_v3_original_discover_dimension()
    finally:
        self.timeout_seconds = original_timeout
        self.retries = original_retries


def _table_worker(pdf_path: str, page_index: int, queue: Any) -> None:
    try:
        import fitz
        from rag_project.utils.text_utils import clean_text
        pdf = fitz.open(pdf_path)
        try:
            page = pdf[page_index]
            finder = getattr(page, "find_tables", None)
            if finder is None:
                queue.put("")
                return
            rendered: list[str] = []
            tables = finder()
            for table in getattr(tables, "tables", []) or []:
                for row in table.extract() or []:
                    line = " | ".join(clean_text(str(cell) if cell is not None else "") for cell in row)
                    if line.strip():
                        rendered.append(line)
                if rendered:
                    rendered.append("")
            queue.put("\n\n".join(rendered).strip())
        finally:
            pdf.close()
    except BaseException as exc:
        try:
            queue.put(f"__BOOKRAG_TABLE_ERROR__:{type(exc).__name__}:{exc}")
        except Exception:
            pass


def _timed_table_extract(page: Any) -> str:
    try:
        text = str(page.get_text("text") or "")
        image_count = len(page.get_images(full=True))
    except Exception:
        return ""
    lowered = text.casefold()
    signal = any(marker in lowered for marker in ("table", "tableau", "tabla", "tab. ", "|", "treatment", "dose", "dosage", "laboratory", "laboratoire", "reference range", "result")) or image_count >= 2
    if not signal:
        return ""
    parent = getattr(page, "parent", None)
    pdf_path = getattr(parent, "name", None)
    page_index = int(getattr(page, "number", -1))
    if not pdf_path or page_index < 0 or not Path(str(pdf_path)).is_file():
        return ""
    timeout = max(3.0, float(os.getenv("RAG_TABLE_TIMEOUT", "20")))
    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_table_worker, args=(str(pdf_path), page_index, queue), daemon=True)
    try:
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)
            return ""
        try:
            result = queue.get(timeout=0.5)
        except Exception:
            return ""
        if isinstance(result, str) and result.startswith("__BOOKRAG_TABLE_ERROR__:"):
            return ""
        return str(result or "")
    finally:
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.app.production_rag import ProductionRAGSystem
        from rag_project.ingestion.state_store import IngestionStateStore
        from rag_project.embeddings.embedding_service import EmbeddingService
        from rag_project.parsing.pdf_extractor import PDFExtractor

        if not hasattr(ProductionRAGSystem, "_runtime_v3_original_ingest_file"):
            ProductionRAGSystem._runtime_v3_original_ingest_file = ProductionRAGSystem.ingest_file
            ProductionRAGSystem.ingest_file = _stable_production_ingest
        if not hasattr(ProductionRAGSystem, "_runtime_v3_original_clear"):
            ProductionRAGSystem._runtime_v3_original_clear = ProductionRAGSystem.clear_pdf_data
            ProductionRAGSystem.clear_pdf_data = _safe_clear
        if not hasattr(ProductionRAGSystem, "_runtime_v3_original_answer"):
            ProductionRAGSystem._runtime_v3_original_answer = ProductionRAGSystem.answer
            ProductionRAGSystem.answer = _locked_answer

        # Older revisions exposed a heavier health_report() method.  The
        # current production base class no longer guarantees that method, so
        # the runtime installer must not dereference it unconditionally.
        # Always expose the lightweight non-blocking health contract, and keep
        # a legacy alias when an original implementation actually exists.
        if not hasattr(ProductionRAGSystem, "_runtime_v3_detailed_health_report"):
            original_health_report = getattr(ProductionRAGSystem, "health_report", None)
            if callable(original_health_report):
                ProductionRAGSystem._runtime_v3_detailed_health_report = original_health_report
            ProductionRAGSystem.health_report_fast = _health_report_fast
            ProductionRAGSystem.health_report = _health_report_fast
        elif not hasattr(ProductionRAGSystem, "health_report_fast"):
            ProductionRAGSystem.health_report_fast = _health_report_fast

        if not hasattr(IngestionStateStore, "_runtime_v3_original_transition"):
            IngestionStateStore._runtime_v3_original_transition = IngestionStateStore.transition_document_state
            IngestionStateStore.transition_document_state = _guard_transition
        if not hasattr(IngestionStateStore, "_runtime_v3_original_get_all_documents"):
            IngestionStateStore._runtime_v3_original_get_all_documents = IngestionStateStore.get_all_documents
            IngestionStateStore.get_all_documents = _enriched_documents
        if not hasattr(IngestionStateStore, "_runtime_v3_original_get_document"):
            IngestionStateStore._runtime_v3_original_get_document = IngestionStateStore.get_document
            IngestionStateStore.get_document = _enriched_document
        if not hasattr(EmbeddingService, "_runtime_v3_original_discover_dimension"):
            EmbeddingService._runtime_v3_original_discover_dimension = EmbeddingService.discover_dimension
            EmbeddingService.discover_dimension = _bounded_discover_dimension
        PDFExtractor._runtime_v3_original_extract_tables = getattr(PDFExtractor, "_extract_tables", None)
        PDFExtractor._extract_tables = staticmethod(_timed_table_extract)
        _INSTALLED = True
