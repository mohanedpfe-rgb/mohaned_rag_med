from __future__ import annotations

from pathlib import Path

import fitz


class DocumentClassifier:
    @staticmethod
    def classify(pdf_path: str | Path) -> dict:
        pdf = fitz.open(str(pdf_path))
        try:
            page_count = pdf.page_count
            text_pages = 0
            image_heavy_pages = 0
            table_heavy_pages = 0
            total_characters = 0

            sample_pages = list(range(page_count))

            for page_index in sample_pages:
                page = pdf[page_index]
                text = page.get_text("text")
                if text and len(text.strip()) > 50:
                    text_pages += 1
                    total_characters += len(text)
                image_count = len(page.get_images())
                if image_count > 2:
                    image_heavy_pages += 1

                if any(token in (text or "").lower() for token in ("table", "figure", "caption")):
                    table_heavy_pages += 1

            if page_count == 0:
                doc_type = "empty"
            elif text_pages / max(page_count, 1) > 0.75:
                doc_type = "text_based"
            elif text_pages / max(page_count, 1) > 0.35:
                doc_type = "mixed"
            else:
                doc_type = "scanned_or_ocr_required"

            return {
                "file_name": Path(pdf_path).name,
                "page_count": page_count,
                "document_type": doc_type,
                "text_pages": text_pages,
                "image_heavy_pages": image_heavy_pages,
                "table_heavy_pages": table_heavy_pages,
                "approx_text_chars": total_characters,
                "ocr_required": doc_type in {"scanned_or_ocr_required", "mixed"},
                "sample_pages": sample_pages,
                "classification_scope": "all_pages",
            }
        finally:
            pdf.close()
