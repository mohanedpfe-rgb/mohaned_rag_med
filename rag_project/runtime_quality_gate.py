from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TERMINAL_STAGES = {
    "READY",
    "COMPLETED",
    "DEGRADED_LEXICAL",
    "FAILED",
    "FAILED_EXTRACTION",
    "FAILED_OCR",
    "FAILED_EMBEDDING",
    "FAILED_INDEXING",
    "QUARANTINED",
}
_ACTIVE_STAGES = {
    "RUNNING",
    "DISCOVERED",
    "VALIDATING",
    "EXTRACTING",
    "OCR",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "VALIDATING_INDEX",
    "INTERRUPTED",
    "RECOVERING",
}


def _lease_is_valid(record: dict[str, Any] | None) -> bool:
    if not record or not record.get("lease_owner"):
        return False
    expires = record.get("lease_expires_at")
    if not expires:
        return False
    try:
        return datetime.fromisoformat(str(expires).replace("Z", "+00:00")) > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def _recover_stale_building_records(vector_store: Any, state_store: Any, logger: Any | None = None) -> dict[str, int]:
    """Remove orphaned BUILDING records left by crashes while preserving live leases."""
    removed_vector = 0
    removed_lexical = 0
    document_versions: set[tuple[str, str]] = set()
    try:
        records = vector_store.collection.get(
            where={"index_state": "BUILDING"},
            include=["metadatas"],
        )
        for metadata in records.get("metadatas", []) or []:
            meta = vector_store._coerce_metadata(metadata)
            document_id = str(meta.get("document_id") or "")
            version_id = str(meta.get("version_id") or "")
            if not document_id or not version_id:
                continue
            record = state_store.get_document(document_id)
            if record and _lease_is_valid(record) and str(record.get("status", "")).upper() in _ACTIVE_STAGES:
                continue
            document_versions.add((document_id, version_id))
        for document_id, version_id in document_versions:
            before = vector_store.count()
            vector_store.delete_version(document_id, version_id)
            removed_vector += max(0, before - vector_store.count())
    except Exception:
        if logger:
            logger.exception("Failed to recover stale BUILDING vector records")

    try:
        with sqlite3.connect(vector_store.lexical_database) as connection:
            rows = connection.execute(
                "SELECT id, metadata FROM lexical_documents WHERE index_state='BUILDING'"
            ).fetchall()
            stale_ids: list[str] = []
            for item_id, raw_metadata in rows:
                try:
                    metadata = json.loads(raw_metadata or "{}")
                except json.JSONDecodeError:
                    metadata = {}
                document_id = str(metadata.get("document_id") or "")
                if not document_id:
                    continue
                record = state_store.get_document(document_id)
                if record and _lease_is_valid(record) and str(record.get("status", "")).upper() in _ACTIVE_STAGES:
                    continue
                stale_ids.append(str(item_id))
            if stale_ids:
                connection.executemany(
                    "DELETE FROM lexical_documents WHERE id=?",
                    [(item_id,) for item_id in stale_ids],
                )
                removed_lexical = len(stale_ids)
    except Exception:
        if logger:
            logger.exception("Failed to recover stale BUILDING lexical records")
    return {"vector_records_removed": removed_vector, "lexical_records_removed": removed_lexical}


def _safe_table_worker(pdf_path: str, page_index: int, queue: Any) -> None:
    """Table worker with type-safe cell coercion."""
    try:
        import fitz
        from rag_project.utils.text_utils import clean_text

        pdf = fitz.open(pdf_path)
        try:
            page = pdf[page_index]
            finder = getattr(page, "find_tables", None)
            if finder is None:
                queue.put((True, ""))
                return
            tables = finder()
            rendered: list[str] = []
            for table in getattr(tables, "tables", []) or []:
                rows = table.extract()
                for row in rows or []:
                    line = " | ".join(clean_text(str(cell) if cell is not None else "") for cell in row)
                    if line.strip():
                        rendered.append(line)
                if rendered:
                    rendered.append("")
            queue.put((True, "\n".join(rendered).strip()))
        finally:
            pdf.close()
    except BaseException as exc:
        try:
            queue.put((False, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass


def _safe_ocr_worker(pdf_path: str, page_index: int, queue: Any, render_scale: int, max_render_pixels: int) -> None:
    """OCR worker honors the extractor's OCR decision and forces OCR when selected."""
    try:
        import fitz
        from rag_project.ocr.ocr_service import OCRService

        service = OCRService(
            use_rapidocr=True,
            lazy_init=False,
            render_scale=render_scale,
            max_render_pixels=max_render_pixels,
        )
        pdf = fitz.open(pdf_path)
        try:
            page = pdf[page_index]
            text, confidence = service.ocr_page_object(page, page_index, force=True)
        finally:
            pdf.close()
        queue.put((True, text or "", confidence))
    except BaseException as exc:
        try:
            queue.put((False, f"{type(exc).__name__}: {exc}", None))
        except Exception:
            pass


def _sanitize_evidence(text: str) -> str:
    """Defense-in-depth source-data sanitization without altering normal medical prose."""
    if not text:
        return text
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    control = {ord(char): None for char in normalized if ord(char) < 32 and char not in "\n\t\r"}
    normalized = normalized.translate(control)
    suspicious = (
        "ignore previous instructions",
        "ignore all instructions",
        "disregard previous instructions",
        "override system prompt",
        "reveal system prompt",
        "developer mode",
        "admin override",
        "jailbreak",
    )
    lines: list[str] = []
    for line in normalized.splitlines():
        folded = " ".join(line.casefold().split())
        role_prefix = folded.startswith(("assistant:", "system:", "developer:", "instruction:", "user:"))
        if role_prefix or any(marker in folded for marker in suspicious):
            lines.append("[REDACTED: source instruction-like text]")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def _quarantine_processed_failure(system: Any, pdf_path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """Guarantee failed files end up in failed_dir even when the source was already moved."""
    if str(result.get("status", "")).lower() != "failed":
        return result
    source = Path(pdf_path)
    processed = Path(system.settings.processed_dir) / source.name
    candidates = [source, processed]
    failed = Path(system.settings.failed_dir) / source.name
    failed.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        if not candidate.exists() or candidate.resolve() == failed.resolve():
            continue
        tmp: Path | None = None
        try:
            tmp = failed.with_name(f".{failed.name}.{os.getpid()}.{time.time_ns()}.part")
            if candidate.stat().st_dev == failed.parent.stat().st_dev:
                os.replace(candidate, tmp)
            else:
                shutil.copy2(candidate, tmp)
                candidate.unlink()
            os.replace(tmp, failed)
            break
        except OSError:
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return result


def _guard_transition(self: Any, document_id: str, new_stage: str, **values: Any) -> None:
    """Reject terminal/indexing transitions when the document lease was lost."""
    stage = str(new_stage).upper()
    if stage in _TERMINAL_STAGES or stage in {"INDEXING", "VALIDATING_INDEX"}:
        record = self.get_document(document_id)
        if not _lease_is_valid(record):
            raise RuntimeError(f"Document lease is not valid for {document_id}; refusing state commit at {stage}.")
    return self._original_runtime_quality_transition(document_id, new_stage, **values)


def _safe_set_version_state(self: Any, document_id: str, version_id: str, state: str) -> None:
    """Serialize cross-store visibility transitions."""
    lock = getattr(self, "_transaction_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._transaction_lock = lock
    with lock:
        return self._original_runtime_quality_set_version_state(document_id, version_id, state)


def _safe_ingest_file(self: Any, pdf_path: str | Path):
    try:
        self.state_store.recover_stale_documents()
    except Exception:
        self.logger.exception("Stale ingestion recovery failed before %s", pdf_path)
    try:
        recovery = _recover_stale_building_records(self.vector_store, self.state_store, self.logger)
        if any(recovery.values()):
            self.logger.info("Recovered stale index records: %s", recovery)
    except Exception:
        self.logger.exception("Stale index recovery failed before %s", pdf_path)
    result = self._original_runtime_quality_ingest_file(pdf_path)
    return _quarantine_processed_failure(self, pdf_path, result)


def _safe_init(self: Any, *args: Any, **kwargs: Any):
    result = self._original_runtime_quality_init(*args, **kwargs)
    try:
        self.state_store.recover_stale_documents()
        recovery = _recover_stale_building_records(self.vector_store, self.state_store, self.logger)
        if any(recovery.values()):
            self.logger.info("Recovered stale records during startup: %s", recovery)
    except Exception:
        self.logger.exception("Startup index recovery failed")
    return result


def _safe_ingest_directory(self: Any, directory: str | Path | None = None):
    """Serialize overlapping directory-ingestion calls so future registries cannot be clobbered."""
    lock = getattr(self, "_directory_ingest_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._directory_ingest_lock = lock
    with lock:
        return self._original_runtime_quality_ingest_directory(directory)


def install() -> None:
    from rag_project.app import rag_system as rag_module
    from rag_project.app.rag_system import RAGSystem
    from rag_project.ingestion.state_store import IngestionStateStore
    import rag_project.runtime_hardening as hardening
    from rag_project.storage.vector_store import VectorStore

    if not hasattr(hardening, "_quality_gate_original_table_worker"):
        hardening._quality_gate_original_table_worker = hardening._run_table_worker
        hardening._run_table_worker = _safe_table_worker

    if not hasattr(hardening, "_quality_gate_original_ocr_worker"):
        hardening._quality_gate_original_ocr_worker = hardening._run_ocr_worker
        hardening._run_ocr_worker = _safe_ocr_worker

    if not hasattr(IngestionStateStore, "_original_runtime_quality_transition"):
        IngestionStateStore._original_runtime_quality_transition = IngestionStateStore.transition_document_state
        IngestionStateStore.transition_document_state = _guard_transition

    if not hasattr(VectorStore, "_original_runtime_quality_set_version_state"):
        VectorStore._original_runtime_quality_set_version_state = VectorStore.set_version_index_state
        VectorStore.set_version_index_state = _safe_set_version_state

    if not hasattr(RAGSystem, "_original_runtime_quality_ingest_file"):
        RAGSystem._original_runtime_quality_ingest_file = RAGSystem.ingest_file
        RAGSystem.ingest_file = _safe_ingest_file

    if not hasattr(RAGSystem, "_original_runtime_quality_ingest_directory"):
        RAGSystem._original_runtime_quality_ingest_directory = RAGSystem.ingest_directory
        RAGSystem.ingest_directory = _safe_ingest_directory

    if not hasattr(RAGSystem, "_original_runtime_quality_init"):
        RAGSystem._original_runtime_quality_init = RAGSystem.__init__
        RAGSystem.__init__ = _safe_init

    rag_module.sanitize_evidence = _sanitize_evidence
