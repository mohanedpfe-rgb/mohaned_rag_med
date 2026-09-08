from __future__ import annotations

from pathlib import Path
import tempfile

import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class OCRService:
    """Lazy-loading OCR service using RapidOCR ONNX runtime."""

    def __init__(
        self,
        use_rapidocr: bool = True,
        *,
        lazy_init: bool = False,
        render_scale: int = 2,
        enhance_contrast: float = 1.8,
        max_render_pixels: int = 12_000_000,
    ):
        self.use_rapidocr = use_rapidocr
        self.lazy_init = bool(lazy_init)
        self.render_scale = max(1, int(render_scale))
        self.enhance_contrast = max(1.0, float(enhance_contrast))
        self.max_render_pixels = max(1_000_000, int(max_render_pixels))
        self._rapidocr = None
        self._init_attempted = False
        self._init_error: str | None = None
        if use_rapidocr and not lazy_init:
            self._ensure_rapidocr()

    def _ensure_rapidocr(self) -> bool:
        if self._rapidocr is not None:
            return True
        if self._init_attempted:
            return self._rapidocr is not None
        self._init_attempted = True
        if not self.use_rapidocr:
            return False
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapidocr = RapidOCR()
            self._init_error = None
            return True
        except Exception as exc:
            self._init_error = f"RapidOCR init failed: {exc}"
            self._rapidocr = None
            return False

    def unload(self) -> None:
        self._rapidocr = None
        self._init_attempted = False

    @staticmethod
    def should_ocr_page(page: fitz.Page) -> bool:
        text = (page.get_text("text") or "").strip()
        return len(text) < 20

    def _render_pixmap(self, page: fitz.Page) -> fitz.Pixmap:
        matrix = fitz.Matrix(self.render_scale, self.render_scale)
        width = max(1, int(page.rect.width * self.render_scale))
        height = max(1, int(page.rect.height * self.render_scale))
        pixels = width * height
        if pixels > self.max_render_pixels:
            scale = (self.max_render_pixels / float(max(1, page.rect.width * page.rect.height))) ** 0.5
            safe_scale = max(1.0, min(float(self.render_scale), scale))
            matrix = fitz.Matrix(safe_scale, safe_scale)
            width = max(1, int(page.rect.width * safe_scale))
            height = max(1, int(page.rect.height * safe_scale))
            pixels = width * height
        if pixels > self.max_render_pixels:
            raise RuntimeError(
                f"OCR render refused: page would require {pixels:,} pixels; "
                f"limit is {self.max_render_pixels:,}."
            )
        return page.get_pixmap(matrix=matrix, alpha=False)

    def ocr_page(self, pdf_path: str | Path, page_index: int) -> tuple[str, float | None]:
        if not self._ensure_rapidocr():
            raise RuntimeError(self._init_error or "No OCR engine available; install rapidocr-onnxruntime.")
        with tempfile.TemporaryDirectory(prefix="rag-ocr-") as directory:
            image_path = Path(directory) / f"page-{page_index + 1}.png"
            pdf = fitz.open(str(pdf_path))
            try:
                page = pdf[page_index]
                if not self.should_ocr_page(page):
                    return "", None
                pix = self._render_pixmap(page)
                pix.save(str(image_path))
            finally:
                pdf.close()
            return self._recognize(image_path)

    def ocr_page_object(
        self,
        page: fitz.Page,
        page_index: int,
        *,
        force: bool = False,
    ) -> tuple[str, float | None]:
        if not self._ensure_rapidocr():
            raise RuntimeError(self._init_error or "No OCR engine available; install rapidocr-onnxruntime.")
        if not force and not self.should_ocr_page(page):
            return "", None
        with tempfile.TemporaryDirectory(prefix="rag-ocr-") as directory:
            image_path = Path(directory) / f"page-{page_index + 1}.png"
            pix = self._render_pixmap(page)
            pix.save(str(image_path))
            return self._recognize(image_path)

    def _recognize(self, image_path: Path) -> tuple[str, float | None]:
        if self._rapidocr is None:
            raise RuntimeError(self._init_error or "No OCR engine available; install rapidocr-onnxruntime.")
        result, _ = self._rapidocr(str(image_path))
        if not result:
            with Image.open(image_path) as image:
                enhanced = ImageOps.autocontrast(ImageOps.grayscale(image))
                enhanced = ImageEnhance.Contrast(enhanced).enhance(self.enhance_contrast)
                enhanced = enhanced.filter(ImageFilter.SHARPEN)
                enhanced.save(image_path)
            result, _ = self._rapidocr(str(image_path))
        if result:
            text = "\n".join(item[1] for item in result if item and len(item) > 1)
            confidence = sum(float(item[2]) for item in result if len(item) > 2) / max(len(result), 1)
            return text.strip(), float(confidence)
        raise RuntimeError("OCR produced no text for this page.")
