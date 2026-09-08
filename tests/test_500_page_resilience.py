from __future__ import annotations

import sys
import types
from pathlib import Path

import fitz

from rag_project.parsing.pdf_extractor import PDFExtractor


def test_500_page_pdf_is_fully_walked(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "synthetic-500-pages.pdf"
    document = fitz.open()
    for index in range(500):
        page = document.new_page()
        page.insert_text((72, 72), f"Synthetic medical document page {index + 1}. Hypertension treatment context.")
    document.save(pdf_path)
    document.close()

    fake_pymupdf4llm = types.SimpleNamespace(
        to_markdown=lambda pdf, pages=None: pdf.load_page(pages[0]).get_text("text") if pages else ""
    )
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake_pymupdf4llm)

    extractor = PDFExtractor(state_store=None, ocr_enabled=False)
    pages = list(extractor.extract_iter(pdf_path, "synthetic-500"))

    assert len(pages) == 500
    assert [page.page_number for page in pages] == list(range(1, 501))
    assert pages[0].text
    assert pages[-1].text
