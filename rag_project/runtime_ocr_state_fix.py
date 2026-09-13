from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rag_project.parsing.pdf_extractor import PDFExtractor

        original = getattr(PDFExtractor, "extract_iter", None)
        if not callable(original) or getattr(original, "_ocr_state_fix", False):
            _INSTALLED = True
            return

        def extract_iter(self: Any, pdf_path: Any, document_id: str | None = None) -> Iterator[Any]:
            for extraction in original(self, pdf_path, document_id):
                status = str(getattr(extraction, "ocr_status", "") or "").casefold()
                if status in {"success", "completed"}:
                    extraction.extraction_method = "ocr"
                    metadata = getattr(extraction, "metadata", None)
                    if isinstance(metadata, dict):
                        metadata["extraction_method"] = "ocr"
                        metadata["ocr_status"] = "completed"
                yield extraction

        extract_iter._ocr_state_fix = True
        extract_iter.__name__ = getattr(original, "__name__", "extract_iter")
        extract_iter.__qualname__ = "PDFExtractor.extract_iter"
        PDFExtractor.extract_iter = extract_iter
        _INSTALLED = True


__all__ = ["install"]
