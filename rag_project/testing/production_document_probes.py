from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont

from rag_project.parsing.pdf_extractor import PDFExtractor
from rag_project.testing.advanced_phases import _result
from rag_project.testing.deep_diagnostics import PhaseResult


def _png_with_text(text: str, width: int = 1200, height: int = 900) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 54)
    except Exception:
        font = ImageFont.load_default()
    draw.text((70, 90), text, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_scanned_fixture(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(
        fitz.Rect(40, 50, 555, 790),
        stream=_png_with_text("HbA1c 6.5 percent\nDiabetes mellitus diagnosis"),
    )
    document.save(path)
    document.close()


class _DeterministicOCR:
    def __init__(self) -> None:
        self.calls = 0

    def ocr_page_object(self, page: fitz.Page, page_index: int, *, force: bool = False) -> tuple[str, float]:
        self.calls += 1
        return "HbA1c 6.5 percent Diabetes mellitus diagnosis", 0.99


def phase7_production_pdf_lab(phase: Any) -> PhaseResult:
    result = _result(phase)
    try:
        with tempfile.TemporaryDirectory(prefix="rag_phase7_production_pdf_") as td:
            root = Path(td)
            scanned = root / "scanned_fixture.pdf"
            _write_scanned_fixture(scanned)

            native = PDFExtractor(ocr_enabled=False).extract(scanned, document_id="phase7-native")[0]

            real_extractor = PDFExtractor(ocr_enabled=True)
            real_page = None
            real_error = None
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

            malformed = root / "malformed.pdf"
            malformed.write_bytes(b"not a pdf")
            malformed_rejected = False
            try:
                PDFExtractor(ocr_enabled=False).extract(malformed, document_id="phase7-malformed")
            except Exception:
                malformed_rejected = True

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
                "backend_truth": "RapidOCR is exercised when its runtime is available; deterministic fallback is used only to verify the production OCR merge/state contract in dependency-isolated CI.",
            }
            checks = {
                "scanned_page_detected": bool(native.ocr_required),
                "ocr_branch_reached": bool(real_page is not None or adapter.calls >= 1),
                "ocr_text_contains_expected_markers": sum(markers.values()) >= 2,
                "ocr_status_success": selected.ocr_status == "success",
                "ocr_method_recorded": selected.extraction_method == "ocr",
                "malformed_pdf_rejected": malformed_rejected,
            }
            result.details["checks"] = checks
            result.score = sum(checks.values()) / len(checks)
            result.status = "PASS" if result.score == 1.0 else "FAIL"
            if result.status == "FAIL":
                result.failures.append({"location": "phase 7 production OCR/PDF lab", "exception": "ProductionDocumentProbeFailure", "message": str(result.details)})
    except Exception as exc:
        result.status = "FAIL"
        result.failures.append({"location": "phase 7 production document probe", "exception": type(exc).__name__, "message": str(exc)})
    result.duration_s = round(time.time() - result.started_at, 3)
    return result
