from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _guarded_extract_tables(page: Any) -> str:
    """Avoid running PyMuPDF table detection on every page.

    Table detection is comparatively expensive. Only invoke it when the page has a
    plausible table signal: a table/column cue in text or a non-trivial amount of
    image content. Any extractor error is treated as an optional-enrichment miss.
    """
    try:
        text = str(page.get_text("text") or "")
        image_count = len(page.get_images(full=True))
    except Exception:
        return ""

    lowered = text.casefold()
    textual_signal = any(
        marker in lowered
        for marker in (
            "table", "tableau", "tabla", "tab. ", "|", "treatment", "dose", "dosage",
            "laboratory", "laboratoire", "reference range", "result",
        )
    )
    image_signal = image_count >= 2
    if not textual_signal and not image_signal:
        return ""

    finder = getattr(page, "find_tables", None)
    if finder is None:
        return ""
    try:
        from rag_project.utils.text_utils import clean_text

        rendered: list[str] = []
        tables = finder()
        for table in getattr(tables, "tables", []) or []:
            rows = table.extract()
            for row in rows or []:
                line = " | ".join(
                    clean_text(str(cell) if cell is not None else "")
                    for cell in row
                )
                if line.strip():
                    rendered.append(line)
            if rendered:
                rendered.append("")
        return "\n\n".join(rendered).strip()
    except (RuntimeError, ValueError, AttributeError, TypeError):
        return ""


def _safe_health_probe(self: Any, force: bool = False) -> bool:
    """Keep Ollama health probes cached even when callers request force=True."""
    now = time.monotonic()
    last = float(getattr(self, "_runtime_v2_health_probe_at", 0.0) or 0.0)
    cached = getattr(self, "_ollama_available", None)
    if cached is not None and now - last < 10.0:
        return bool(cached)
    original = self._runtime_v2_original_health_probe
    result = bool(original(force=False))
    self._runtime_v2_health_probe_at = now
    return result


def _disk_preflight(system: Any, pdf_path: Path) -> None:
    """Fail quickly on storage conditions that would otherwise surface late in indexing."""
    for directory in (
        Path(system.settings.vector_db_dir),
        Path(system.settings.ingestion_db_path).parent,
        Path(system.settings.processed_dir),
        Path(system.settings.failed_dir),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".bookrag-write-test-{os.getpid()}"
        try:
            with probe.open("wb") as handle:
                handle.write(b"ok")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Storage preflight failed for {directory}: {exc}") from exc

    usage = shutil.disk_usage(Path(system.settings.vector_db_dir))
    required_floor = max(256 * 1024 * 1024, min(int(pdf_path.stat().st_size * 8), 2 * 1024 * 1024 * 1024))
    if usage.free < required_floor:
        raise RuntimeError(
            f"Insufficient free storage for indexing: {usage.free / (1024**3):.2f} GiB available, "
            f"approximately {required_floor / (1024**3):.2f} GiB safety floor."
        )


def _mark_clean_failure(system: Any, pdf_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Never accept a partial semantic->lexical fallback as a successful full index."""
    if not result.get("degraded"):
        return result
    document_id = result.get("document_id")
    if not document_id:
        return result
    try:
        record = system.state_store.get_document(document_id)
        version_id = str((record or {}).get("content_hash") or result.get("content_hash") or "")
        if version_id:
            system.vector_store.delete_version(document_id, version_id)
        system.state_store.update_document(
            document_id,
            current_stage="FAILED_EMBEDDING",
            status="FAILED_EMBEDDING",
            error=(
                "Semantic embedding became unavailable after semantic indexing had already started; "
                "the partial index was rolled back instead of returning an incomplete lexical index."
            ),
            index_state="FAILED",
        )
        failed = Path(system.settings.failed_dir) / pdf_path.name
        failed.parent.mkdir(parents=True, exist_ok=True)
        processed = Path(system.settings.processed_dir) / pdf_path.name
        if processed.exists() and not failed.exists():
            os.replace(processed, failed)
        elif pdf_path.exists() and pdf_path.resolve() != failed.resolve():
            os.replace(pdf_path, failed)
        return {
            "status": "failed",
            "document_id": document_id,
            "file_name": pdf_path.name,
            "error": (
                "Embedding service failed during semantic indexing. The partial index was rolled back; "
                "start Ollama and retry this PDF."
            ),
            "failure_stage": "FAILED_EMBEDDING",
        }
    except Exception as exc:
        system.logger.exception("Failed to roll back degraded ingestion for %s", pdf_path.name)
        return {
            "status": "failed",
            "document_id": document_id,
            "file_name": pdf_path.name,
            "error": f"Semantic embedding failed and rollback also failed: {exc}",
            "failure_stage": "FAILED_EMBEDDING",
        }


def _safe_ingest_file(self: Any, pdf_path: str | Path) -> dict[str, Any]:
    """Final guard around the stable state machine.

    The stable implementation can intentionally fall back to lexical indexing when
    semantic embedding is unavailable from the very beginning. That is safe. What is
    not safe is switching halfway through a semantic build, because earlier semantic
    batches are then removed and the remaining lexical batches do not cover the whole
    document. This wrapper detects that case and rolls it back instead of reporting a
    misleading success.
    """
    source = Path(pdf_path)
    _disk_preflight(self, source)

    lock = getattr(self, "_runtime_v2_ingest_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._runtime_v2_ingest_lock = lock

    with lock:
        preflight_semantic = True
        try:
            self.embedding_service.discover_dimension()
            self.embedding_startup_error = None
        except Exception as exc:
            preflight_semantic = False
            self.embedding_startup_error = str(exc)
            self.logger.warning(
                "Semantic embedding preflight unavailable; this ingestion will use lexical mode: %s",
                exc,
            )

        result = self._runtime_v2_original_ingest_file(source)
        if preflight_semantic and isinstance(result, dict) and result.get("degraded"):
            return _mark_clean_failure(self, source, result)
        return result


def install() -> None:
    """Install the final stability corrections after every older runtime layer."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from rag_project.app.rag_system import RAGSystem
        from rag_project.embeddings.embedding_service import EmbeddingService
        from rag_project.parsing.pdf_extractor import PDFExtractor

        if not hasattr(PDFExtractor, "_runtime_v2_original_extract_tables"):
            PDFExtractor._runtime_v2_original_extract_tables = PDFExtractor._extract_tables
            PDFExtractor._extract_tables = staticmethod(_guarded_extract_tables)

        if not hasattr(EmbeddingService, "_runtime_v2_original_health_probe"):
            EmbeddingService._runtime_v2_original_health_probe = EmbeddingService._check_ollama_available
            EmbeddingService._runtime_v2_health_probe_at = 0.0
            EmbeddingService._check_ollama_available = _safe_health_probe

        if not hasattr(RAGSystem, "_runtime_v2_original_ingest_file"):
            RAGSystem._runtime_v2_original_ingest_file = RAGSystem.ingest_file
            RAGSystem.ingest_file = _safe_ingest_file

        _INSTALLED = True
