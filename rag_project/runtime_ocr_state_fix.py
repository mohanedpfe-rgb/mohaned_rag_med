from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def _mark_ocr(extraction: Any) -> Any:
    status = str(getattr(extraction, "ocr_status", "") or "").casefold()
    if status in {"success", "completed"}:
        extraction.extraction_method = "ocr"
        metadata = getattr(extraction, "metadata", None)
        if isinstance(metadata, dict):
            metadata["extraction_method"] = "ocr"
            metadata["ocr_status"] = "completed"
    return extraction


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.ingestion.state_store import IngestionStateStore
        from rag_project.parsing.pdf_extractor import PDFExtractor

        original_record_page = getattr(IngestionStateStore, "record_page", None)
        if callable(original_record_page) and not getattr(original_record_page, "_ocr_state_record_fix", False):
            def record_page(self: Any, extraction: Any, *args: Any, **kwargs: Any):
                return original_record_page(self, _mark_ocr(extraction), *args, **kwargs)
            record_page._ocr_state_record_fix = True
            record_page.__name__ = getattr(original_record_page, "__name__", "record_page")
            IngestionStateStore.record_page = record_page

        original = getattr(PDFExtractor, "extract_iter", None)
        if callable(original) and not getattr(original, "_ocr_state_fix", False):
            def extract_iter(self: Any, pdf_path: Any, document_id: str | None = None) -> Iterator[Any]:
                for extraction in original(self, pdf_path, document_id):
                    yield _mark_ocr(extraction)
            extract_iter._ocr_state_fix = True
            extract_iter.__name__ = getattr(original, "__name__", "extract_iter")
            extract_iter.__qualname__ = "PDFExtractor.extract_iter"
            PDFExtractor.extract_iter = extract_iter

        _INSTALLED = True


__all__ = ["install"]
