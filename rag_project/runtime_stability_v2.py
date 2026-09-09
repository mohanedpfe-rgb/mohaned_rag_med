from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Iterator


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def _guarded_extract_tables(page: Any) -> str:
    """Avoid running PyMuPDF table detection on every page."""
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
    """Fail quickly on storage conditions that would otherwise surface late."""
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
    required_floor = max(
        256 * 1024 * 1024,
        min(int(pdf_path.stat().st_size * 8), 2 * 1024 * 1024 * 1024),
    )
    if usage.free < required_floor:
        raise RuntimeError(
            f"Insufficient free storage for indexing: {usage.free / (1024**3):.2f} GiB available, "
            f"approximately {required_floor / (1024**3):.2f} GiB safety floor."
        )


def _ocr_worker_force(pdf_path: str, page_index: int, queue: Any, render_scale: int, max_render_pixels: int) -> None:
    """Killable OCR worker used only for pages that actually require OCR."""
    try:
        from rag_project.ocr.ocr_service import OCRService

        service = OCRService(
            use_rapidocr=True,
            lazy_init=False,
            render_scale=render_scale,
            max_render_pixels=max_render_pixels,
        )
        import fitz

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


def _timed_ocr(pdf_path: Path, page_index: int, render_scale: int, max_render_pixels: int, timeout: float) -> tuple[str, float | None]:
    """Run OCR in a killable child process so a native/ONNX stall cannot freeze ingestion."""
    timeout = max(5.0, float(timeout))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_ocr_worker_force,
        args=(str(pdf_path), int(page_index), queue, int(render_scale), int(max_render_pixels)),
    )
    process.daemon = True
    process.start()
    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)
            raise RuntimeError(f"OCR hard timeout after {timeout:.1f}s on page {page_index + 1}")
        try:
            item = queue.get(timeout=0.5)
        except Exception as exc:
            raise RuntimeError(
                f"OCR worker exited with code {process.exitcode} and returned no result for page {page_index + 1}"
            ) from exc
        if not item or not item[0]:
            raise RuntimeError(str(item[1] if len(item) > 1 else "OCR worker failed"))
        return str(item[1] or ""), item[2]
    finally:
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass


class _TimeoutOCRProxy:
    """Adapter preserving OCRService's public surface while adding a hard timeout."""

    def __init__(self, original: Any, pdf_path: Path, timeout: float, max_render_pixels: int):
        self._original = original
        self._pdf_path = pdf_path
        self._timeout = timeout
        self._max_render_pixels = max_render_pixels
        self.render_scale = int(getattr(original, "render_scale", 2))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def ocr_page_object(self, page: Any, page_index: int, *, force: bool = False):
        if not force and not getattr(self._original, "should_ocr_page", lambda _: False)(page):
            return "", None
        return _timed_ocr(
            self._pdf_path,
            page_index,
            self.render_scale,
            self._max_render_pixels,
            self._timeout,
        )


def _safe_extract_iter_with_ocr_timeout(
    self: Any,
    pdf_path: str | Path,
    document_id: str | None = None,
) -> Iterator[Any]:
    original_service = getattr(self, "ocr_service", None)
    timeout = max(5.0, float(getattr(self, "ocr_timeout_seconds", os.getenv("RAG_OCR_TIMEOUT", "45"))))
    max_pixels = max(1_000_000, int(getattr(self, "ocr_max_render_pixels", os.getenv("RAG_OCR_MAX_PIXELS", "12000000"))))
    if original_service is None:
        yield from self._runtime_v2_original_extract_iter(pdf_path, document_id)
        return
    self.ocr_service = _TimeoutOCRProxy(original_service, Path(pdf_path), timeout, max_pixels)
    try:
        yield from self._runtime_v2_original_extract_iter(pdf_path, document_id)
    finally:
        self.ocr_service = original_service


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
    """Final guard around the stable state machine."""
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
    """Install final ingestion stability corrections after every older runtime layer."""
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

        if not hasattr(PDFExtractor, "_runtime_v2_original_extract_iter"):
            PDFExtractor._runtime_v2_original_extract_iter = PDFExtractor.extract_iter
            PDFExtractor.extract_iter = _safe_extract_iter_with_ocr_timeout

        if not hasattr(RAGSystem, "_runtime_v2_original_ingest_file"):
            RAGSystem._runtime_v2_original_ingest_file = RAGSystem.ingest_file
            RAGSystem.ingest_file = _safe_ingest_file

        _INSTALLED = True
