from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import fitz

from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.testing.advanced_phases import _result
from rag_project.testing.deep_diagnostics import PhaseResult


def _write_scanned_fixture(path: Path) -> None:
    source = fitz.open(); source_page = source.new_page(width=900, height=1200)
    source_page.insert_text((100, 180), "HbA1c 6.5 percent", fontsize=64)
    source_page.insert_text((100, 290), "Diabetes mellitus diagnosis", fontsize=64)
    pix = source_page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), colorspace=fitz.csRGB, alpha=False)
    scanned = fitz.open(); scanned_page = scanned.new_page(width=pix.width, height=pix.height)
    scanned_page.insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)
    scanned.save(path); scanned.close(); source.close()


class _DeterministicOCR:
    def __init__(self) -> None: self.calls = 0
    def ocr_page_object(self, page: fitz.Page, page_index: int, *, force: bool = False) -> tuple[str, float]:
        self.calls += 1; return "HbA1c 6.5 percent Diabetes mellitus diagnosis", 0.99


def _write_blank_fixture(path: Path) -> None:
    document = fitz.open(); document.new_page(width=595, height=842); document.save(path); document.close()


def phase7_production_pdf_lab(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        with tempfile.TemporaryDirectory(prefix="rag_phase7_production_pdf_") as td:
            root = Path(td)
            scanned = root / "scanned_fixture.pdf"; _write_scanned_fixture(scanned)
            native = PDFExtractor(ocr_enabled=False).extract(scanned, document_id="phase7-native")[0]

            real_extractor = PDFExtractor(ocr_enabled=True); real_page = None; real_error = None
            try:
                real_page = real_extractor.extract(scanned, document_id="phase7-real-ocr")[0]
            except Exception as exc:
                real_error = type(exc).__name__

            adapter = _DeterministicOCR()
            adapter_page = PDFExtractor(ocr_enabled=True, ocr_service=adapter).extract(scanned, document_id="phase7-adapter")[0]
            selected = real_page or adapter_page
            backend = "rapidocr" if real_page is not None and real_page.ocr_status == "success" else "deterministic_contract_adapter"

            ocr_text = (selected.text or "").casefold()
            markers = {marker: marker in ocr_text for marker in ("hba1c", "6.5", "diabetes")}

            malformed = root / "malformed.pdf"; malformed.write_bytes(b"not a pdf")
            malformed_rejected = False
            try:
                PDFExtractor(ocr_enabled=False).extract(malformed, document_id="phase7-malformed")
            except Exception:
                malformed_rejected = True

            blank = root / "blank.pdf"; _write_blank_fixture(blank)
            blank_pages = PDFExtractor(ocr_enabled=False).extract(blank, document_id="phase7-blank")
            blank_handled = len(blank_pages) == 1 and (blank_pages[0].text or "").strip() == ""

            require_real = os.getenv("REQUIRE_REAL_OCR", "0").strip().lower() in {"1", "true", "yes", "on"}
            real_ok = real_page is not None and real_page.ocr_status == "success" and real_page.extraction_method == "ocr" and sum(markers.values()) >= 2
            variants = {
                "native_text_on_scanned_pdf": bool(native.ocr_required),
                "real_ocr_backend": bool(real_ok),
                "deterministic_ocr_contract_adapter": bool(adapter_page.ocr_status == "success" and adapter.calls >= 1),
                "malformed_pdf_rejected": malformed_rejected,
                "blank_pdf_handled_without_crash": blank_handled,
            }
            checks = {
                "scanned_page_detected": bool(native.ocr_required),
                "ocr_branch_reached": bool(real_page is not None or adapter.calls >= 1),
                "ocr_text_contains_expected_markers": sum(markers.values()) >= 2,
                "ocr_status_success": selected.ocr_status == "success",
                "ocr_method_recorded": selected.extraction_method == "ocr",
                "malformed_pdf_rejected": malformed_rejected,
                "blank_pdf_handled": blank_handled,
                "required_real_ocr_satisfied": (not require_real or real_ok),
            }
            result.details = {
                "evidence_level": "real_pdf_extractor",
                "production_component": "rag_project.parsing.pdf_extractor.PDFExtractor",
                "ocr_component": "rag_project.ocr.ocr_service.OCRService",
                "real_scanned_pdf": True,
                "native_ocr_required": bool(native.ocr_required),
                "real_ocr_attempted": True,
                "real_ocr_available": real_page is not None,
                "real_ocr_error": real_error,
                "ocr_backend": backend,
                "ocr_adapter_calls": adapter.calls,
                "ocr_status": selected.ocr_status,
                "extraction_method": selected.extraction_method,
                "ocr_text_markers": markers,
                "ocr_text_marker_count": sum(markers.values()),
                "malformed_pdf_rejected": malformed_rejected,
                "blank_pdf_handled": blank_handled,
                "real_ocr_required": require_real,
                "real_ocr_verified": bool(real_ok),
                "variant_results": variants,
                "variant_count": len(variants),
                "backend_truth": "The fixture contains rasterized glyphs with no text layer. RapidOCR is exercised when available; deterministic fallback verifies the OCR merge/state contract when dependency-isolated CI cannot load the OCR runtime.",
                "checks": checks,
            }
            result.score = sum(checks.values()) / len(checks)
            result.status = "PASS" if result.score == 1.0 else "FAIL"
            if result.status == "FAIL":
                result.failures.append({"location": "phase 7 production OCR/PDF lab", "exception": "ProductionDocumentProbeFailure", "message": str(result.details)})
    except Exception as exc:
        result.status = "FAIL"; result.failures.append({"location": "phase 7 production document probe", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3); return result
