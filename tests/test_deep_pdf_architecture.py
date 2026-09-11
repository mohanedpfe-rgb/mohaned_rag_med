from __future__ import annotations

from rag_project.ingestion.document_models import PageExtraction
from rag_project.intelligence.deep_pdf_contract import install
from rag_project.intelligence.document_structure import DocumentStructureTracker, STRUCTURE_SCHEMA_VERSION
from rag_project.chunking.semantic_chunker import SemanticChunker


install()


def _page(document_id: str, number: int, text: str, *, tables=None, captions=None) -> PageExtraction:
    tables = list(tables or [])
    captions = list(captions or [])
    return PageExtraction(
        document_id=document_id,
        file_name="book.pdf",
        page_index=number - 1,
        page_number=number,
        text=text,
        extraction_method="pdf_text",
        page_type="text_based",
        table_count=len(tables),
        image_count=max(1, len(captions)),
        has_images=bool(captions),
        quality_score=0.9,
        routing_decision="native",
        table_ids=[f"t{number}-{i}" for i in range(len(tables))],
        figure_ids=[f"f{number}-{i}" for i in range(len(captions))],
        table_texts=tables,
        figure_captions=captions,
        ocr_status="not_required",
    )


def test_tracker_keeps_section_identity_across_pages() -> None:
    tracker = DocumentStructureTracker("doc")
    first = tracker.analyze_page(1, "Chapter 3 Respiratory System\n3.1 Ventilation\nA paragraph.")
    second = tracker.analyze_page(2, "The section continues here without a heading.")
    assert first.chapter_id == second.chapter_id
    assert first.section_id == second.section_id
    assert first.parent_id == second.parent_id


def test_chunker_preserves_global_hierarchy_for_tables_and_figures() -> None:
    chunker = SemanticChunker(300, 20)
    pages = [
        _page("doc", 1, "Chapter 3 Respiratory System\n3.1 Ventilation\nText one."),
        _page(
            "doc",
            2,
            "Text two.\n[TABLE]\nAge | Male | Female\n20 | 1 | 2\n21 | 3 | 4",
            tables=["Age | Male | Female\n20 | 1 | 2\n21 | 3 | 4"],
            captions=["Figure 4. Ventilation diagram"],
        ),
    ]
    chunks = chunker.chunk_pages(pages)
    canonical = [c for c in chunks if c.representation_type == "canonical"]
    tables = [c for c in chunks if c.representation_type == "table"]
    figures = [c for c in chunks if c.representation_type == "figure_caption"]
    assert canonical and tables and figures
    assert tables[0].metadata["section_id"] == canonical[-1].metadata["section_id"]
    assert figures[0].metadata["section_id"] == canonical[-1].metadata["section_id"]
    assert tables[0].metadata["parent_id"] == canonical[-1].metadata["parent_id"]
    assert figures[0].metadata["parent_id"] == canonical[-1].metadata["parent_id"]
    assert STRUCTURE_SCHEMA_VERSION >= 3
