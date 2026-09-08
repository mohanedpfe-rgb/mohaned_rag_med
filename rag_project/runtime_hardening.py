from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, List

logger = logging.getLogger(__name__)


def _run_markdown_worker(pdf_path: str, page_index: int, queue: Any) -> None:
    try:
        import fitz
        import pymupdf4llm

        pdf = fitz.open(pdf_path)
        try:
            value = pymupdf4llm.to_markdown(pdf, pages=[page_index])
        finally:
            pdf.close()
        queue.put((True, value or ""))
    except BaseException as exc:  # worker boundary
        queue.put((False, f"{type(exc).__name__}: {exc}"))


def _run_table_worker(pdf_path: str, page_index: int, queue: Any) -> None:
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
            for table in getattr(tables, "tables", []):
                rows = table.extract()
                for row in rows:
                    line = " | ".join(clean_text(cell or "") for cell in row)
                    if line.strip():
                        rendered.append(line)
                if rendered:
                    rendered.append("")
            queue.put((True, "\n".join(rendered).strip()))
        finally:
            pdf.close()
    except BaseException as exc:
        queue.put((False, f"{type(exc).__name__}: {exc}"))


def _run_ocr_worker(pdf_path: str, page_index: int, queue: Any, render_scale: int) -> None:
    try:
        from rag_project.ocr.ocr_service import OCRService

        service = OCRService(use_rapidocr=True, lazy_init=False, render_scale=render_scale)
        text, confidence = service.ocr_page(pdf_path, page_index)
        queue.put((True, text or "", confidence))
    except BaseException as exc:
        queue.put((False, f"{type(exc).__name__}: {exc}", None))


def _timed_process(fn: Any, args: tuple[Any, ...], timeout: float) -> tuple[bool, Any, str | None]:
    """Run unsafe PDF/ONNX work outside the ingestion process so a hard timeout can kill it."""
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=fn, args=(*args, queue))
    process.daemon = True
    process.start()
    process.join(max(0.1, float(timeout)))
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(1.0)
        return False, None, f"hard timeout after {timeout:.1f}s"
    try:
        item = queue.get_nowait()
    except Exception:
        return False, None, f"worker exited with code {process.exitcode} and no result"
    if not item[0]:
        return False, None, str(item[1])
    return True, item[1:], None


def _safe_extract_iter(self: Any, pdf_path: str | Path, document_id: str | None = None) -> Iterator[Any]:
    """Replacement page iterator: native text stays local; pymupdf4llm/OCR/tables are killable."""
    import fitz
    from rag_project.ingestion.document_models import PageExtraction
    from rag_project.utils.text_utils import clean_text, extract_page_number, split_paragraphs

    pdf_file = Path(pdf_path)
    document_id = document_id or __import__("uuid").uuid4().hex
    page_timeout = max(3.0, float(getattr(self, "page_timeout_seconds", 30.0)))
    table_timeout = max(1.0, float(getattr(self, "table_timeout_seconds", 5.0)))
    ocr_timeout = max(5.0, float(getattr(self, "ocr_timeout_seconds", 45.0)))
    ocr_enabled = bool(getattr(self, "ocr_enabled", False))
    pdf = fitz.open(str(pdf_file))
    try:
        cached_pages = {}
        if getattr(self, "state_store", None):
            cached_pages = {item["page_number"]: item for item in self.state_store.get_pages(document_id)}
        total = pdf.page_count
        for index in range(total):
            physical_page = index + 1
            # Cooperative cancellation: an interrupted/cancelled document must not write new pages.
            if getattr(self, "state_store", None):
                state = self.state_store.get_document(document_id) or {}
                status = str(state.get("status", "")).upper()
                if status in {"CANCELLED", "INTERRUPTED"}:
                    logger.info("Stopping extraction for %s at page %d: %s", document_id, physical_page, status)
                    return
            cached = cached_pages.get(physical_page)
            if cached and cached.get("extraction_status") == "COMPLETED" and cached.get("text"):
                text = str(cached["text"])
                yield PageExtraction(
                    document_id=document_id,
                    file_name=pdf_file.name,
                    page_index=index,
                    page_number=physical_page,
                    text=text,
                    extraction_method=cached.get("extraction_method") or "cached",
                    ocr_status=cached.get("ocr_status") or "not_required",
                    blocks=[p.strip() for p in split_paragraphs(text) if p.strip()],
                    source_path=str(pdf_file),
                    metadata={"physical_page": physical_page, "cache_hit": True},
                )
                continue

            page = pdf[index]
            native_text = page.get_text("blocks", sort=True)
            raw_native = "\n\n".join(
                str(block[4]).strip() for block in native_text if len(block) > 4 and str(block[4]).strip()
            ) or page.get_text("text", sort=True)
            text = clean_text(raw_native)
            extraction_method = "pdf_text"

            ok, payload, error = _timed_process(
                _run_markdown_worker,
                (str(pdf_file), index),
                page_timeout,
            )
            if ok and payload:
                markdown = clean_text(str(payload[0]))
                if markdown.strip():
                    text = markdown
                    extraction_method = "pymupdf4llm"
            elif error:
                logger.warning("Page %d pymupdf4llm fallback: %s", physical_page, error)

            word_count = len(text.split())
            image_count = len(page.get_images(full=True))
            page_area = max(1.0, float(page.rect.width * page.rect.height))
            char_density = len(text) / page_area
            image_coverage = 0.0
            try:
                for image in page.get_images(full=True):
                    for rect in page.get_image_rects(image[0]):
                        image_coverage += (rect.width * rect.height) / page_area
            except Exception:
                pass
            ocr_required = len(text) < 120 or word_count < 25 or char_density < float(getattr(self, "ocr_min_char_density", 0.0008))
            if image_count and image_coverage >= float(getattr(self, "ocr_image_coverage_threshold", 0.55)) and word_count < 30:
                ocr_required = True

            table_text = ""
            if image_count or "table" in text.casefold() or "|" in text:
                ok, payload, error = _timed_process(_run_table_worker, (str(pdf_file), index), table_timeout)
                if ok and payload:
                    table_text = str(payload[0] or "").strip()
                elif error:
                    logger.debug("Page %d table extraction skipped: %s", physical_page, error)
            if table_text:
                text = clean_text(f"{text}\n\n{table_text}")

            ocr_status = "not_required"
            confidence = None
            metadata: dict[str, Any] = {
                "physical_page": physical_page,
                "printed_page_number": extract_page_number(text),
                "word_count": len(text.split()),
                "char_count": len(text),
                "image_count": image_count,
                "image_coverage": round(image_coverage, 4),
                "char_density": round(char_density, 6),
                "page_timeout_seconds": page_timeout,
            }
            if ocr_required:
                if not ocr_enabled:
                    ocr_status = "skipped_disabled"
                    metadata["ocr_required_reason"] = "low_native_text_density"
                else:
                    ok, payload, error = _timed_process(
                        _run_ocr_worker,
                        (str(pdf_file), index, int(getattr(self.ocr_service, "render_scale", 2))),
                        ocr_timeout,
                    )
                    if ok and payload:
                        ocr_text = clean_text(str(payload[0] or ""))
                        confidence = payload[1]
                        threshold = float(getattr(self, "ocr_confidence_threshold", 0.55))
                        if ocr_text and (confidence is None or float(confidence) >= threshold):
                            if ocr_text.casefold() not in text.casefold():
                                text = clean_text(f"{text}\n\n{ocr_text}").strip()
                            extraction_method = "ocr" if extraction_method == "pdf_text" else f"{extraction_method}+ocr"
                            ocr_status = "success"
                        else:
                            ocr_status = "skipped_low_confidence" if confidence is not None else "skipped_no_result"
                    else:
                        ocr_status = "failed"
                        metadata["ocr_error"] = error

            extraction = PageExtraction(
                document_id=document_id,
                file_name=pdf_file.name,
                page_index=index,
                page_number=physical_page,
                text=text,
                extraction_method=extraction_method,
                ocr_required=ocr_required,
                ocr_status=ocr_status,
                ocr_confidence=float(confidence) if confidence is not None else None,
                page_type="image_heavy" if image_count and word_count < 60 else ("text_based" if word_count >= 25 else "scanned_or_ocr_required"),
                image_count=image_count,
                table_count=1 if table_text else 0,
                has_images=bool(image_count),
                blocks=[p.strip() for p in split_paragraphs(text) if p.strip()],
                metadata=metadata,
                source_path=str(pdf_file),
            )
            if getattr(self, "state_store", None) and self.state_store.get_document(document_id):
                self.state_store.upsert_page(
                    document_id,
                    physical_page,
                    extraction_status="COMPLETED" if text else "FAILED",
                    ocr_status=ocr_status,
                    extraction_method=extraction_method,
                    text=text,
                    processing_error=metadata.get("ocr_error"),
                    checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
                self.state_store.update_document(document_id, current_stage="EXTRACTING", current_page=physical_page)
            yield extraction
    finally:
        pdf.close()


def _safe_lexical_search(self: Any, query: str, n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return self._as_query_result([], [], [])
    tokens = set(self._lexical_tokens(query))
    if not tokens:
        return self._as_query_result([], [], [])
    with sqlite3.connect(self.lexical_database) as connection:
        records = connection.execute(
            "SELECT id, document, metadata, tokens FROM lexical_documents WHERE index_state='READY'"
        ).fetchall()
    corpus = [json.loads(row[3]) for row in records]
    document_count = len(corpus)
    average_length = max(1.0, sum(len(row_tokens) for row_tokens in corpus) / max(1, document_count))
    document_frequency = {token: sum(token in row_tokens for row_tokens in corpus) for token in tokens}
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row, row_tokens in zip(records, corpus, strict=True):
        meta = self._coerce_metadata(json.loads(row[2]))
        if str(meta.get("index_state", "READY")).upper() != "READY" or not self._metadata_matches(meta, where):
            continue
        length = max(1, len(row_tokens))
        score = 0.0
        for token in tokens:
            frequency = row_tokens.count(token)
            if not frequency:
                continue
            df = document_frequency[token]
            idf = __import__("math").log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            score += idf * (frequency * 2.2) / (frequency + 1.2 * (0.75 + 0.25 * length / average_length))
        if score > 0:
            ranked.append((score, {"id": str(row[0]), "document": str(row[1]), "metadata": meta}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked = ranked[: max(1, int(n_results))]
    return self._as_query_result(
        [item[1]["id"] for item in ranked],
        [item[1]["document"] for item in ranked],
        [item[1]["metadata"] for item in ranked],
        [1.0 / (1.0 + item[0]) for item in ranked],
    )


def _safe_classify(pdf_path: str | Path) -> dict[str, Any]:
    import fitz

    path = Path(pdf_path)
    pdf = fitz.open(str(path))
    try:
        count = pdf.page_count
        if not count:
            sample = []
        else:
            sample_count = min(16, count)
            sample = sorted({round(i * (count - 1) / max(1, sample_count - 1)) for i in range(sample_count)})
        text_pages = image_heavy_pages = table_heavy_pages = chars = 0
        for index in sample:
            page = pdf[index]
            text = page.get_text("text") or ""
            if len(text.strip()) > 50:
                text_pages += 1
                chars += len(text)
            if len(page.get_images(full=True)) > 2:
                image_heavy_pages += 1
            if any(token in text.casefold() for token in ("table", "figure", "caption")):
                table_heavy_pages += 1
        sample_len = max(1, len(sample))
        ratio = text_pages / sample_len
        doc_type = "empty" if count == 0 else "text_based" if ratio > 0.75 else "mixed" if ratio > 0.35 else "scanned_or_ocr_required"
        return {
            "file_name": path.name,
            "page_count": count,
            "document_type": doc_type,
            "text_pages": text_pages,
            "image_heavy_pages": image_heavy_pages,
            "table_heavy_pages": table_heavy_pages,
            "approx_text_chars": chars,
            "ocr_required": doc_type in {"scanned_or_ocr_required", "mixed"},
            "sample_pages": sample,
            "classification_scope": "stratified_sample",
        }
    finally:
        pdf.close()


def _safe_chunk_page_batches(self: Any, pages: Iterable[Any], batch_size: int = 16) -> Iterator[list[Any]]:
    buffer: list[Any] = []
    for page in pages:
        buffer.append(page)
        if len(buffer) >= max(1, int(batch_size)):
            chunks = self.chunk_pages(buffer)
            if chunks:
                yield chunks
            buffer = []
    if buffer:
        chunks = self.chunk_pages(buffer)
        if chunks:
            yield chunks


def _safe_rag_clear(self: Any) -> None:
    """Do not clear stores while ingestion workers still have a chance to write stale data."""
    cancel = getattr(self, "cancel_all_ingests", None)
    if callable(cancel):
        cancel()
    futures = list(getattr(self, "_ingest_futures", {}).values()) if hasattr(getattr(self, "_ingest_futures", None), "values") else list(getattr(self, "_ingest_futures", []) or [])
    for future in futures:
        try:
            future.cancel()
        except Exception:
            pass
    for future in futures:
        try:
            future.result(timeout=60)
        except Exception:
            pass
    lock = getattr(self, "_index_write_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._index_write_lock = lock
    with lock:
        self.vector_store.clear_all()
        self.state_store.clear_all()


def install() -> None:
    """Install defensive runtime patches once at application startup."""
    if getattr(install, "_done", False):
        return
    from rag_project.parsing.pdf_extractor import PDFExtractor
    from rag_project.storage.vector_store import VectorStore
    from rag_project.ingestion.document_classifier import DocumentClassifier
    from rag_project.chunking.semantic_chunker import SemanticChunker
    from rag_project.app.rag_system import RAGSystem

    if not hasattr(PDFExtractor, "_original_extract_iter"):
        PDFExtractor._original_extract_iter = PDFExtractor.extract_iter
        PDFExtractor.extract_iter = _safe_extract_iter
        original_init = PDFExtractor.__init__

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("ocr_enabled", False)
            original_init(self, *args, **kwargs)
            self.page_timeout_seconds = float(os.getenv("RAG_PAGE_TIMEOUT", "30"))
            self.table_timeout_seconds = float(os.getenv("RAG_TABLE_TIMEOUT", "5"))
            self.ocr_timeout_seconds = float(os.getenv("RAG_OCR_TIMEOUT", "45"))

        PDFExtractor.__init__ = init

    if not hasattr(VectorStore, "_original_search_lexical"):
        VectorStore._original_search_lexical = VectorStore.search_lexical
        VectorStore.search_lexical = _safe_lexical_search

    if not hasattr(DocumentClassifier, "_original_classify"):
        DocumentClassifier._original_classify = DocumentClassifier.classify
        DocumentClassifier.classify = staticmethod(_safe_classify)

    if not hasattr(SemanticChunker, "_original_chunk_page_batches"):
        SemanticChunker._original_chunk_page_batches = SemanticChunker.chunk_page_batches
        SemanticChunker.chunk_page_batches = _safe_chunk_page_batches

    if not hasattr(RAGSystem, "_original_clear_pdf_data"):
        original_clear = getattr(RAGSystem, "clear_pdf_data", None)
        if original_clear:
            RAGSystem._original_clear_pdf_data = original_clear
            RAGSystem.clear_pdf_data = _safe_rag_clear

    install._done = True
    logger.info("Runtime hardening installed")
