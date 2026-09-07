from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.ingestion.document_models import PageExtraction


def test_semantic_chunker_creates_chunks():
    pages = [
        PageExtraction(
            document_id="doc-1",
            file_name="sample.pdf",
            page_index=0,
            page_number=1,
            text="This is a paragraph about retrieval and generation. It has enough content to form a first chunk.",
            extraction_method="pdf_text",
            ocr_required=False,
            page_type="text_based",
        ),
        PageExtraction(
            document_id="doc-1",
            file_name="sample.pdf",
            page_index=1,
            page_number=2,
            text="This is a second paragraph about citations and evaluation. It must be separate from the first chunk.",
            extraction_method="pdf_text",
            ocr_required=False,
            page_type="text_based",
        ),
    ]
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=25)
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) >= 1
    assert all(chunk.text for chunk in chunks)


def test_semantic_chunker_handles_large_paragraphs_and_empty_page_list():
    long_text = " ".join(["word"] * 300)
    pages = [
        PageExtraction(
            document_id="doc-2",
            file_name="long.pdf",
            page_index=0,
            page_number=1,
            text=long_text,
            extraction_method="pdf_text",
            ocr_required=False,
            page_type="text_based",
        )
    ]
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_pages(pages)
    assert chunks
    assert all(len(chunk.text) <= 230 for chunk in chunks)
    assert chunker.chunk_pages([]) == []


def test_pdf_extractor_and_chunker_support_iterable_batches(tmp_path: Path):
    pdf_path = tmp_path / "streamed.pdf"
    document = fitz.open()
    for index in range(3):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {index + 1} streamed extraction content.")
    document.save(str(pdf_path))
    document.close()

    from rag_project.parsing.pdf_extractor import PDFExtractor

    pages = PDFExtractor().extract_iter(pdf_path, "streamed-doc")
    batches = list(SemanticChunker(chunk_size=200).chunk_page_batches(pages, batch_size=1))
    assert len(batches) == 3
    assert all(batch and batch[0].doc_id == "streamed-doc" for batch in batches)
    assert [batch[0].chunk_index for batch in batches] == [0, 1, 2]


def test_extract_pdf_classification_works_for_generated_document(tmp_path: Path):
    pdf_path = tmp_path / "generated.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Chapter 1\nThis is a generated PDF for testing the RAG pipeline.")
    document.save(str(pdf_path))
    document.close()

    from rag_project.ingestion.document_classifier import DocumentClassifier
    classification = DocumentClassifier.classify(pdf_path)
    assert classification["page_count"] == 1
    assert classification["document_type"] in {"text_based", "mixed"}


def test_image_heavy_pdf_is_extracted_with_ocr_metadata(tmp_path: Path):
    image_path = tmp_path / "embedded.png"
    Image.new("RGB", (800, 600), "white").save(image_path)
    pdf_path = tmp_path / "image-heavy.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_image(page.rect, filename=str(image_path))
    document.save(str(pdf_path))
    document.close()

    from rag_project.parsing.pdf_extractor import PDFExtractor

    extracted = list(PDFExtractor().extract_iter(pdf_path, "image-doc"))
    assert len(extracted) == 1
    assert extracted[0].has_images is True
    assert extracted[0].ocr_required is True
    assert extracted[0].ocr_status in {"success", "failed"}


def test_table_like_pdf_preserves_text_without_crashing(tmp_path: Path):
    pdf_path = tmp_path / "table-like.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "TABLE\nDrug | Dose | Frequency\nMetformin | 500 mg | Twice daily",
    )
    document.save(str(pdf_path))
    document.close()

    from rag_project.parsing.pdf_extractor import PDFExtractor

    extracted = list(PDFExtractor().extract_iter(pdf_path, "table-doc"))
    assert "Metformin" in extracted[0].text
    assert extracted[0].text


def test_chunks_preserve_table_and_figure_evidence_types():
    pages = [
        PageExtraction(
            document_id="structured-doc",
            file_name="structured.pdf",
            page_index=0,
            page_number=1,
            text="A table and figure describe the treatment plan.",
            extraction_method="pdf_text",
            table_count=1,
            image_count=1,
            has_images=True,
        )
    ]
    chunks = SemanticChunker(chunk_size=200).chunk_pages(pages)
    assert chunks[0].metadata["evidence_types"] == ["figure", "table", "text"]
