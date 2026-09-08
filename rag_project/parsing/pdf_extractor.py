from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fitz

logger = logging.getLogger(__name__)

from rag_project.ingestion.document_models import PageExtraction
from rag_project.ocr.ocr_service import OCRService
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.utils.text_utils import clean_text, extract_page_number, split_paragraphs


class PDFExtractor:
    """High-quality PDF extractor using PyMuPDF + pymupdf4llm with smart optional OCR."""

    def __init__(
        self,
        state_store: IngestionStateStore | None = None,
        *,
        ocr_enabled: bool = True,
        ocr_confidence_threshold: float = 0.55,
        ocr_min_char_density: float = 0.001,
        ocr_image_coverage_threshold: float = 0.55,
        ocr_service: OCRService | None = None,
    ):
        self.ocr_enabled = bool(ocr_enabled)
        self.ocr_confidence_threshold = max(0.0, float(ocr_confidence_threshold))
        self.ocr_min_char_density = max(0.0, float(ocr_min_char_density))
        self.ocr_image_coverage_threshold = max(0.0, float(ocr_image_coverage_threshold))
        self.ocr_service = ocr_service or OCRService(use_rapidocr=True, lazy_init=True)
        self.state_store = state_store

    @staticmethod
    def _normalize_for_dedupe(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").casefold()).strip()

    @classmethod
    def _looks_duplicate(cls, existing: str, candidate: str) -> bool:
        if not existing or not candidate:
            return False
        normalized_existing = cls._normalize_for_dedupe(existing)
        normalized_candidate = cls._normalize_for_dedupe(candidate)
        if not normalized_existing or not normalized_candidate:
            return False
        if normalized_existing == normalized_candidate:
            return True
        if normalized_candidate in normalized_existing or normalized_existing in normalized_candidate:
            return True
        existing_tokens = set(re.findall(r"\w+", normalized_existing))
        candidate_tokens = set(re.findall(r"\w+", normalized_candidate))
        if not existing_tokens or not candidate_tokens:
            return False
        overlap = len(existing_tokens & candidate_tokens) / max(len(existing_tokens | candidate_tokens), 1)
        return overlap >= 0.8

    @classmethod
    def _merge_native_and_ocr_text(cls, native_text: str, ocr_text: str) -> str:
        native = [part.strip() for part in split_paragraphs(native_text or "") if part.strip()]
        ocr = [part.strip() for part in split_paragraphs(ocr_text or "") if part.strip()]
        merged: list[str] = []
        seen: list[str] = []
        for part in [*native, *ocr]:
            normalized = cls._normalize_for_dedupe(part)
            if not normalized:
                continue
            if any(cls._looks_duplicate(existing, part) for existing in seen):
                continue
            merged.append(part)
            seen.append(part)
        return clean_text("\n\n".join(merged)) if merged else clean_text(native_text or ocr_text or "")

    @staticmethod
    def _assess_page_ocr_need(
        page: fitz.Page,
        text: str,
        *,
        min_char_count: int = 120,
        min_alpha_count: int = 40,
        min_word_count: int = 25,
        image_coverage_threshold: float = 0.55,
        image_heavy_word_limit: int = 30,
        min_char_density: float = 0.0008,
    ) -> dict[str, Any]:
        """Precise assessment of whether a page actually needs OCR."""
        word_count = len((text or "").split())
        char_count = len(text or "")
        alpha_count = sum(
            1
            for character in text
            if unicodedata.category(character).startswith(("L", "N"))
        )
        image_count, image_coverage_pixels = PDFExtractor._safe_image_stats(page)
        has_images = image_count > 0
        page_area = page.rect.width * page.rect.height if page.rect.width and page.rect.height else 1.0
        heavy_image_pixels = image_coverage_pixels
        image_coverage = heavy_image_pixels / page_area if page_area > 0 else 0.0
        char_density = char_count / page_area if page_area > 0 else 0.0

        too_little_text = char_count < min_char_count
        sparse_letters = alpha_count < min_alpha_count and word_count < min_word_count
        image_dominated = image_coverage > image_coverage_threshold and word_count < image_heavy_word_limit
        garbled_hint = False
        if char_count >= min_char_count:
            printable_ratio = sum(
                1 for c in text if c.isprintable() or c in "\n\r\t"
            ) / max(len(text), 1)
            garbled_hint = printable_ratio < 0.85
        low_density = char_density < min_char_density and word_count < min_word_count

        reasons: list[str] = []
        if too_little_text:
            reasons.append(f"few_chars ({char_count})")
        if sparse_letters:
            reasons.append(f"sparse_alphanum (alpha={alpha_count}, words={word_count})")
        if image_dominated:
            reasons.append(f"image_heavy (coverage={image_coverage:.2f}, words={word_count})")
        if garbled_hint:
            reasons.append("garbled_text_layer")
        if low_density:
            reasons.append(f"low_char_density ({char_density:.5f})")

        ocr_required = bool(reasons)

        page_type = "text_based"
        if has_images and word_count < 60:
            page_type = "image_heavy"
        elif ocr_required:
            page_type = "scanned_or_ocr_required"
        elif word_count < 100:
            page_type = "sparse_text"

        return {
            "ocr_required": ocr_required,
            "reasons": reasons,
            "page_type": page_type,
            "word_count": word_count,
            "char_count": char_count,
            "alpha_count": alpha_count,
            "image_count": image_count,
            "has_images": has_images,
            "image_coverage": float(image_coverage),
            "char_density": float(char_density),
        }

    @staticmethod
    def _safe_image_stats(page: fitz.Page) -> tuple[int, float]:
        """Return image count and covered pixels without aborting on stale page objects."""
        try:
            images = page.get_images(full=True)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Skipping image inspection on page %s: %s", getattr(page, "number", "?"), exc)
            return 0, 0.0
        coverage = 0.0
        for img in images:
            try:
                xref = img[0]
                for box in page.get_image_rects(xref):
                    coverage += box.width * box.height
            except Exception:
                continue
        return len(images), coverage

    @staticmethod
    def _load_page(pdf: fitz.Document, index: int) -> fitz.Page:
        try:
            return pdf.load_page(index)
        except Exception:
            return pdf[index]

    def extract(self, pdf_path: str | Path, document_id: str | None = None) -> list[PageExtraction]:
        return list(self.extract_iter(pdf_path, document_id))

    def extract_iter(
        self, pdf_path: str | Path, document_id: str | None = None
    ) -> Iterator[PageExtraction]:
        pdf_file = Path(pdf_path)
        document_id = document_id or str(uuid.uuid4())
        pdf = None
        try:
            try:
                pdf = fitz.open(str(pdf_file))
            except RuntimeError as exc:
                raise RuntimeError(f"{pdf_file.name}: failed to open PDF: {exc}") from exc
            cached_pages = {
                item["page_number"]: item
                for item in self.state_store.get_pages(document_id)
            } if self.state_store else {}
            last_progress_page = 0
            for index in range(pdf.page_count):
                page_number = index + 1
                if self.state_store:
                    cached = cached_pages.get(page_number)
                    if cached and cached["extraction_status"] == "COMPLETED" and cached.get("text"):
                        text = cached["text"]
                        if page_number % 4 == 0 or page_number == pdf.page_count:
                            self.state_store.update_document(
                                document_id, current_stage="EXTRACTING", current_page=page_number
                            )
                            last_progress_page = page_number
                        yield PageExtraction(
                            document_id=document_id,
                            file_name=pdf_file.name,
                            page_index=index,
                            page_number=page_number,
                            text=text,
                            extraction_method=cached.get("extraction_method") or "cached",
                            ocr_status=cached.get("ocr_status", "not_required"),
                            blocks=[p.strip() for p in split_paragraphs(text) if p.strip()],
                            source_path=str(pdf_file),
                        )
                        continue
                try:
                    page = self._load_page(pdf, index)
                    import pymupdf4llm

                    raw_text = pymupdf4llm.to_markdown(pdf, pages=[index])
                    page = self._load_page(pdf, index)
                    if not raw_text or not raw_text.strip():
                        raw_text = self._extract_page_text(page)
                except Exception:
                    page = self._load_page(pdf, index)
                    raw_text = self._extract_page_text(page)
                text = clean_text(raw_text)
                blocks = [p.strip() for p in split_paragraphs(text) if p.strip()]
                assessment = self._assess_page_ocr_need(
                    page,
                    text,
                    image_coverage_threshold=self.ocr_image_coverage_threshold,
                    min_char_density=self.ocr_min_char_density,
                )
                page_number_from_text = extract_page_number(text) or page_number
                tables_text = self._extract_tables(page)
                if tables_text:
                    text = clean_text(f"{text}\n\n{tables_text}")
                    blocks = [p.strip() for p in split_paragraphs(text) if p.strip()]
                    assessment["word_count"] = len(text.split())
                    assessment["char_count"] = len(text)

                ocr_required_flag = assessment["ocr_required"]
                page_type = assessment["page_type"]
                image_count = assessment["image_count"]
                has_images = assessment["has_images"]

                extraction = PageExtraction(
                    document_id=document_id,
                    file_name=pdf_file.name,
                    page_index=index,
                    page_number=page_number_from_text,
                    text=text,
                    extraction_method="pdf_text",
                    ocr_required=ocr_required_flag,
                    ocr_status="not_required",
                    ocr_confidence=None,
                    page_type=page_type,
                    image_count=image_count,
                    table_count=len(tables_text.split("\n\n")) if tables_text else 0,
                    has_images=has_images,
                    blocks=blocks,
                    metadata={
                        "word_count": assessment["word_count"],
                        "char_count": assessment["char_count"],
                        "alpha_count": assessment["alpha_count"],
                        "image_coverage": round(assessment["image_coverage"], 4),
                        "char_density": round(assessment["char_density"], 6),
                    },
                    source_path=str(pdf_file),
                )

                if ocr_required_flag:
                    if not self.ocr_enabled:
                        extraction.ocr_status = "skipped_disabled"
                        extraction.metadata["ocr_disabled"] = True
                        extraction.metadata["ocr_reasons"] = assessment["reasons"]
                    else:
                        try:
                            ocr_text, confidence = self.ocr_service.ocr_page_object(
                                page, index, force=True
                            )
                            if not ocr_text:
                                extraction.ocr_status = "skipped_no_result"
                                extraction.ocr_required = False
                                extraction.metadata["ocr_skipped"] = (
                                    "Page already has a text layer above the OCR threshold or OCR produced no text."
                                )
                            elif confidence is not None and confidence < self.ocr_confidence_threshold:
                                extraction.ocr_status = "skipped_low_confidence"
                                extraction.metadata["ocr_confidence"] = float(confidence)
                                extraction.metadata["ocr_threshold"] = self.ocr_confidence_threshold
                            else:
                                ocr_text = clean_text(ocr_text)
                                native_text = extraction.text.strip()
                                merged_text = self._merge_native_and_ocr_text(native_text, ocr_text)
                                final_text = merged_text or ocr_text or native_text
                                if final_text != native_text:
                                    extraction.text = final_text
                                    extraction.extraction_method = "ocr"
                                    extraction.blocks = [
                                        p.strip() for p in split_paragraphs(extraction.text) if p.strip()
                                    ]
                                extraction.ocr_required = True
                                extraction.ocr_status = "success"
                                extraction.ocr_confidence = confidence
                                extraction.metadata["ocr_char_count"] = len(ocr_text)
                                extraction.metadata["ocr_reasons"] = assessment["reasons"]
                        except Exception as exc:  # pragma: no cover - fallback path
                            extraction.ocr_status = "failed"
                            extraction.metadata["ocr_error"] = str(exc)
                            extraction.metadata["quality_warning"] = (
                                "Page is marked as OCR-required but OCR could not run."
                            )
                            extraction.metadata["ocr_reasons"] = assessment["reasons"]
                if self.state_store and self.state_store.get_document(document_id):
                    self.state_store.upsert_page(
                        document_id,
                        page_number,
                        extraction_status="COMPLETED" if extraction.text else "FAILED",
                        ocr_status=extraction.ocr_status,
                        extraction_method=extraction.extraction_method,
                        text=extraction.text,
                        processing_error=extraction.metadata.get("ocr_error"),
                        checksum=hashlib.sha256(extraction.text.encode("utf-8")).hexdigest(),
                    )
                    if (
                        page_number % 4 == 0 or page_number == pdf.page_count
                    ) and page_number != last_progress_page:
                        self.state_store.update_document(
                            document_id, current_stage="EXTRACTING", current_page=page_number
                        )
                        last_progress_page = page_number
                yield extraction
        except Exception as exc:
            if pdf is None:
                raise RuntimeError(f"{pdf_file.name}: failed to open PDF: {exc}") from exc
            raise
        finally:
            if pdf is not None:
                pdf.close()

    @staticmethod
    def _extract_page_text(page: fitz.Page) -> str:
        """Extract in visual reading order instead of trusting PDF object order."""
        blocks = page.get_text("blocks", sort=True)
        if blocks:
            return "\n\n".join(
                str(block[4]).strip()
                for block in blocks
                if len(block) > 4 and str(block[4]).strip()
            )
        return page.get_text("text", sort=True)

    @staticmethod
    def _extract_tables(page: fitz.Page) -> str:
        """Add table cell text when supported by the installed PyMuPDF version."""
        find_tables = getattr(page, "find_tables", None)
        if find_tables is None:
            return ""
        try:
            tables = find_tables()
            rendered: list[str] = []
            for table in getattr(tables, "tables", []):
                rows = table.extract()
                lines = [" | ".join(clean_text(cell or "") for cell in row) for row in rows]
                rendered.append("\n".join(line for line in lines if line.strip()))
            return "\n\n".join(item for item in rendered if item.strip())
        except (RuntimeError, ValueError, AttributeError):
            return ""

    def extract_markdown(self, pdf_path: str | Path) -> str:
        """Optional structure-preserving export using pymupdf4llm when installed."""
        try:
            import pymupdf4llm
        except ImportError:
            return ""
        return pymupdf4llm.to_markdown(str(pdf_path))
