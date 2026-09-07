from __future__ import annotations

import unicodedata
import uuid
import hashlib
from collections.abc import Iterator
from pathlib import Path

import fitz

from rag_project.ingestion.document_models import PageExtraction
from rag_project.ocr.ocr_service import OCRService
from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.utils.text_utils import clean_text, extract_page_number, split_paragraphs


class PDFExtractor:
    def __init__(self, state_store: IngestionStateStore | None = None):
        self.ocr_service = OCRService()
        self.state_store = state_store

    def extract(self, pdf_path: str | Path, document_id: str | None = None) -> list[PageExtraction]:
        return list(self.extract_iter(pdf_path, document_id))

    def extract_iter(
        self, pdf_path: str | Path, document_id: str | None = None
    ) -> Iterator[PageExtraction]:
        pdf_file = Path(pdf_path)
        document_id = document_id or str(uuid.uuid4())
        pdf = None
        try:
            pdf = fitz.open(str(pdf_file))
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
                page = pdf[index]
                try:
                    import pymupdf4llm

                    raw_text = pymupdf4llm.to_markdown(pdf, pages=[index])
                    if not raw_text or not raw_text.strip():
                        raw_text = self._extract_page_text(page)
                except Exception:
                    raw_text = self._extract_page_text(page)
                text = clean_text(raw_text)
                blocks = [p.strip() for p in split_paragraphs(text) if p.strip()]
                word_count = len((text or "").split())
                char_count = len(text or "")
                page_number = extract_page_number(text) or page_number
                image_count = len(page.get_images())
                has_images = image_count > 0
                alpha_count = sum(
                    1
                    for character in text
                    if unicodedata.category(character).startswith(("L", "N"))
                )
                total_area = page.rect.width * page.rect.height if has_images else 0.0
                heavy_image_pixels = 0.0
                if has_images:
                    for img in page.get_images(full=True):
                        try:
                            xref = img[0]
                            for bb in page.get_image_rects(xref):
                                heavy_image_pixels += bb.width * bb.height
                        except Exception:
                            pass
                image_coverage = (
                    heavy_image_pixels / total_area if total_area > 0 else 0.0
                )
                ocr_required = (
                    char_count < 120
                    or (alpha_count < 40 and word_count < 25)
                    or (image_coverage > 0.55 and word_count < 30)
                )
                tables_text = self._extract_tables(page)
                if tables_text:
                    text = clean_text(f"{text}\n\n{tables_text}")
                    blocks = [p.strip() for p in split_paragraphs(text) if p.strip()]
                    word_count = len(text.split())
                    char_count = len(text)
                page_type = "text_based"
                if has_images and word_count < 60:
                    page_type = "image_heavy"
                elif word_count < 100:
                    page_type = "scanned_or_ocr_required"

                extraction = PageExtraction(
                    document_id=document_id,
                    file_name=pdf_file.name,
                    page_index=index,
                    page_number=page_number,
                    text=text,
                    extraction_method="pdf_text" if not ocr_required else "pdf_text_then_ocr",
                    ocr_required=ocr_required,
                    ocr_status="not_required",
                    ocr_confidence=None,
                    page_type=page_type,
                    image_count=image_count,
                    table_count=len(tables_text.split("\n\n")) if tables_text else 0,
                    has_images=has_images,
                    blocks=blocks,
                    metadata={"word_count": word_count, "char_count": char_count},
                    source_path=str(pdf_file),
                )

                if ocr_required:
                    try:
                        ocr_text, confidence = self.ocr_service.ocr_page_object(page, index)
                        if not ocr_text:
                            extraction.ocr_status = "skipped"
                            extraction.ocr_required = False
                            extraction.metadata["ocr_skipped"] = "Page already has a text layer above the OCR threshold."
                        else:
                            ocr_text = clean_text(ocr_text)
                            # Keep selectable PDF text when OCR is clearly worse, but
                            # replace sparse/garbled extraction with the OCR result.
                            if len(ocr_text) > len(extraction.text) or not extraction.text:
                                extraction.text = ocr_text
                                extraction.extraction_method = "ocr"
                                extraction.blocks = [p.strip() for p in split_paragraphs(extraction.text) if p.strip()]
                            extraction.ocr_required = True
                            extraction.ocr_status = "success"
                            extraction.ocr_confidence = confidence
                            extraction.metadata["ocr_char_count"] = len(ocr_text)
                    except Exception as exc:  # pragma: no cover - fallback path
                        extraction.ocr_status = "failed"
                        extraction.metadata["ocr_error"] = str(exc)
                        extraction.metadata["quality_warning"] = "Page is marked as OCR-required but OCR could not run."
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
