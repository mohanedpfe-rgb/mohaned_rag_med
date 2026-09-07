from __future__ import annotations

from pathlib import Path
import tempfile

import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class OCRService:
    def __init__(self, use_rapidocr: bool = True):
        self.use_rapidocr = use_rapidocr
        self._rapidocr = None
        if use_rapidocr:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._rapidocr = RapidOCR()
            except Exception:
                self._rapidocr = None

    @staticmethod
    def should_ocr_page(page: fitz.Page) -> bool:
        """Only run OCR when the page does not already have enough text."""
        text = (page.get_text("text") or "").strip()
        return len(text) < 20

    def ocr_page(self, pdf_path: str | Path, page_index: int) -> tuple[str, float | None]:
        with tempfile.TemporaryDirectory(prefix="rag-ocr-") as directory:
            image_path = Path(directory) / f"page-{page_index + 1}.png"
            pdf = fitz.open(str(pdf_path))
            try:
                page = pdf[page_index]
                if not self.should_ocr_page(page):
                    return "", None
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
                pix.save(str(image_path))
            finally:
                pdf.close()

            return self._recognize(image_path)

    def ocr_page_object(self, page: fitz.Page, page_index: int) -> tuple[str, float | None]:
        """OCR an already-open page to avoid reopening a large PDF per page."""
        if not self.should_ocr_page(page):
            return "", None
        with tempfile.TemporaryDirectory(prefix="rag-ocr-") as directory:
            image_path = Path(directory) / f"page-{page_index + 1}.png"
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            pix.save(str(image_path))
            return self._recognize(image_path)

    def _recognize(self, image_path: Path) -> tuple[str, float | None]:
        if self._rapidocr is None:
            raise RuntimeError("No OCR engine available; install Tesseract or rapidocr-onnxruntime.")
        result, _ = self._rapidocr(str(image_path))
        if not result:
            with Image.open(image_path) as image:
                enhanced = ImageOps.autocontrast(ImageOps.grayscale(image))
                enhanced = ImageEnhance.Contrast(enhanced).enhance(1.8)
                enhanced.filter(ImageFilter.SHARPEN).save(image_path)
            result, _ = self._rapidocr(str(image_path))
        if result:
            text = "\n".join(item[1] for item in result if item and len(item) > 1)
            confidence = sum(item[2] for item in result if len(item) > 2) / max(len(result), 1)
            return text.strip(), float(confidence)
        raise RuntimeError("OCR produced no text for this page.")
