from __future__ import annotations

import re
from typing import Iterator

from rag_project.parsing.pdf_extractor import PDFExtractor


_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?|illustration|plate)\s*\d*\s*[:.\-]?\s*.+$", re.I)


class UniversalPDFExtractor(PDFExtractor):
    """Production wrapper that upgrades OCR pages with searchable table/figure units."""

    @staticmethod
    def _extract_table_blocks(text: str) -> list[str]:
        blocks: list[str] = []
        for match in re.finditer(r"\[TABLE\]\s*\n(.*?)(?=\n\n|$)", text or "", flags=re.I | re.S):
            block = match.group(1).strip()
            if block:
                blocks.append(block)
        return blocks

    @staticmethod
    def _heuristic_ocr_table(text: str) -> str | None:
        rows: list[list[str]] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("[TABLE]"):
                continue
            parts = [part.strip() for part in re.split(r"\t+|\s{2,}", line) if part.strip()]
            if len(parts) >= 2:
                rows.append(parts)
        if len(rows) < 2:
            return None
        width = max(len(row) for row in rows)
        if width < 2:
            return None
        normalized = [row + [""] * (width - len(row)) for row in rows]
        return "\n".join(" | ".join(row) for row in normalized)

    @staticmethod
    def _figure_captions(text: str) -> list[str]:
        captions: list[str] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if _CAPTION_RE.match(line):
                captions.append(line)
        return captions

    def extract_iter(self, pdf_path, document_id=None) -> Iterator:
        for page in super().extract_iter(pdf_path, document_id):
            table_texts = self._extract_table_blocks(page.text)
            if not table_texts and page.page_type == "image_heavy" and page.ocr_status == "success":
                candidate = self._heuristic_ocr_table(page.text)
                if candidate:
                    table_texts = [candidate]
                    page.text = f"{page.text.rstrip()}\n\n[TABLE]\n{candidate}".strip()
                    page.table_count = max(int(page.table_count or 0), 1)
                    if not page.table_ids:
                        page.table_ids = [f"{page.document_id}:p{page.page_number}:table:ocr-1"]
                    page.metadata["ocr_table_recovered"] = True
                    page.metadata["evidence_types"] = sorted(set(page.metadata.get("evidence_types", [])) | {"table"})

            captions = self._figure_captions(page.text)
            if page.has_images and not captions:
                captions = [f"Figure/image on physical page {page.page_number}; no textual caption was available in the PDF."]
            page.metadata["table_texts"] = table_texts
            page.metadata["figure_captions"] = captions
            # Runtime code uses getattr so these fields remain backward compatible,
            # while the dataclass update makes them explicit for new documents.
            page.table_texts = table_texts
            page.figure_captions = captions

            if page.has_images and captions:
                page.metadata["figure_searchable"] = True
            if getattr(self, "state_store", None) is not None:
                self.state_store.upsert_page(
                    page.document_id,
                    int(page.page_number or page.page_index + 1),
                    extraction_status="COMPLETED" if page.text else "FAILED",
                    ocr_status=page.ocr_status,
                    extraction_method=page.extraction_method,
                    text=page.text,
                    processing_error=page.metadata.get("ocr_error"),
                    checksum=None,
                )
            yield page
