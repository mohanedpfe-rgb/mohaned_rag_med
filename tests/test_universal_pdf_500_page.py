from __future__ import annotations

import fitz

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.parsing.pdf_extractor import PDFExtractor


def test_500_page_pdf_streams_all_pages_and_preserves_page_metadata(tmp_path):
    path = tmp_path / "500-pages.pdf"
    document = fitz.open()
    for page_no in range(1, 501):
        page = document.new_page()
        page.insert_text((40, 60), f"Page {page_no}: Clinical evidence section. Paracetamol 500 mg.")
    document.save(path)
    document.close()

    pages = list(PDFExtractor(ocr_enabled=False).extract_iter(path, "doc-500"))
    assert len(pages) == 500
    assert pages[0].page_number == 1
    assert pages[-1].page_number == 500
    assert all(p.quality_score >= 0 for p in pages)

    chunks = SemanticChunker(chunk_size=300, chunk_overlap=30).chunk_pages(pages[:3])
    assert chunks
    assert all(chunk.page_numbers for chunk in chunks)
    assert all(chunk.metadata.get("document_id") == "doc-500" for chunk in chunks)
    assert all("normalized_text" in chunk.metadata for chunk in chunks)
