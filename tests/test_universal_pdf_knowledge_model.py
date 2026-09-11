from __future__ import annotations

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.configuration.config_i5_16gb import I5_16GB_PRESET
from rag_project.ingestion.document_models import PageExtraction
from rag_project.storage.enhanced_vector_store import EnhancedVectorStore, _unit_mean


def test_i5_profile_enables_ocr_for_scanned_books() -> None:
    assert I5_16GB_PRESET["ocr_enabled"] is True


def test_chunker_emits_durable_structure_context() -> None:
    page = PageExtraction(
        document_id="doc-1",
        file_name="book.pdf",
        page_index=0,
        page_number=1,
        text="# Chapter One\n## Anatomy\nThe heart is a muscular organ.",
        extraction_method="pdf_text",
        page_type="text_based",
        quality_score=0.91,
        ocr_status="not_required",
    )
    chunks = SemanticChunker(chunk_size=500, chunk_overlap=0).chunk_pages([page])
    assert chunks
    first = chunks[0]
    assert "[RAG-STRUCTURE" in first.text
    assert first.metadata["chapter_id"]
    assert first.metadata["section_id"]
    assert first.metadata["parent_id"]
    assert first.metadata["quality_score"] == 0.91


def test_structure_parser_rehydrates_metadata_before_vector_persistence() -> None:
    text = (
        "[RAG-STRUCTURE chapter_id=ch1; chapter=Cardiology; section_id=sec1; section=Heart; "
        "parent_id=par1; quality=0.8750; page_type=image_heavy; ocr_status=success; table_id=; figure_id=]\n"
        "[Chapter: Cardiology - Section: Heart]\nAtrial fibrillation."
    )
    enriched = EnhancedVectorStore._enrich_metadata(
        {"document_id": "doc-1", "page_numbers": [4]}, text, "chunk-1"
    )
    assert enriched["chapter_id"] == "ch1"
    assert enriched["section_id"] == "sec1"
    assert enriched["parent_id"] == "par1"
    assert enriched["quality_score"] == 0.875
    assert enriched["page_type"] == "image_heavy"
    assert enriched["ocr_status"] == "success"
    assert enriched["record_type"] == "chunk"


def test_table_and_figure_chunks_get_first_class_types() -> None:
    table = EnhancedVectorStore._enrich_metadata(
        {"document_id": "doc-1"}, "[TABLE]\nReference ranges", "chunk-table"
    )
    figure = EnhancedVectorStore._enrich_metadata(
        {"document_id": "doc-1"}, "[FIGURE CAPTION]\nFigure 2: cardiac cycle", "chunk-figure"
    )
    assert table["representation_type"] == "table"
    assert table["table_id"]
    assert figure["representation_type"] == "figure_caption"
    assert figure["figure_id"]


def test_hierarchy_centroid_is_unit_normalized() -> None:
    vector = _unit_mean([[3.0, 0.0], [0.0, 4.0]])
    assert len(vector) == 2
    assert abs((vector[0] ** 2 + vector[1] ** 2) - 1.0) < 1e-9
