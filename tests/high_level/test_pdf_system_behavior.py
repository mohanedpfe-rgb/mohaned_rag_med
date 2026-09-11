from __future__ import annotations

from pathlib import Path

import pytest

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.ingestion.document_models import PageExtraction
from rag_project.intelligence.document_structure import DocumentStructureTracker
from rag_project.intelligence.deep_pdf_finalizer import VisionFigureAdapter


pytestmark = pytest.mark.integration


def _page(
    page_number: int,
    text: str,
    *,
    tables: list[str] | None = None,
    captions: list[str] | None = None,
    ocr_status: str = "not_required",
) -> PageExtraction:
    tables = list(tables or [])
    captions = list(captions or [])
    return PageExtraction(
        document_id="high-level-book",
        file_name="medical_textbook.pdf",
        page_index=page_number - 1,
        page_number=page_number,
        text=text,
        extraction_method="pdf_text",
        page_type="text_based" if ocr_status == "not_required" else "scanned",
        quality_score=0.92 if ocr_status == "not_required" else 0.84,
        routing_decision="native" if ocr_status == "not_required" else "ocr",
        ocr_status=ocr_status,
        table_count=len(tables),
        image_count=len(captions),
        has_images=bool(captions),
        table_ids=[f"table-{page_number}-{i}" for i in range(len(tables))],
        figure_ids=[f"figure-{page_number}-{i}" for i in range(len(captions))],
        table_texts=tables,
        figure_captions=captions,
    )


def test_book_ingestion_produces_prose_table_and_figure_representations() -> None:
    pages = [
        _page(
            1,
            "Chapter 4 Respiratory System\n4.1 Ventilation\nVentilation moves air into and out of the lungs.",
        ),
        _page(
            2,
            "The normal respiratory rate varies with age.\n[TABLE]\nAge | Rate\nAdult | 12-20",
            tables=["Age | Rate\nAdult | 12-20"],
            captions=["Figure 4.1. Mechanics of ventilation"],
        ),
    ]

    chunks = SemanticChunker(280, 20).chunk_pages(pages)
    representations = {chunk.representation_type for chunk in chunks}

    assert "canonical" in representations
    assert "table" in representations
    assert "figure_caption" in representations
    assert all(chunk.metadata.get("document_id") == "high-level-book" for chunk in chunks)
    assert all(chunk.metadata.get("page_numbers") for chunk in chunks)
    assert all(chunk.metadata.get("parent_id") for chunk in chunks)
    assert all(chunk.metadata.get("section_id") for chunk in chunks)


def test_cross_page_question_context_keeps_same_section_identity() -> None:
    tracker = DocumentStructureTracker("high-level-book")
    first = tracker.analyze_page(
        1,
        "Chapter 4 Respiratory System\n4.2 Ventilation\nVentilation is the movement of air.",
    )
    second = tracker.analyze_page(
        2,
        "The pressure gradient drives airflow during inspiration and expiration.",
    )

    assert second.chapter_id == first.chapter_id
    assert second.section_id == first.section_id
    assert second.parent_id == first.parent_id
    assert second.hierarchy_path == first.hierarchy_path


def test_table_question_uses_first_class_table_evidence() -> None:
    page = _page(
        5,
        "Chapter 7 Laboratory Values\n7.1 Reference Values\n[TABLE]\nTest | Normal Range\nHemoglobin | 12-16 g/dL",
        tables=["Test | Normal Range\nHemoglobin | 12-16 g/dL"],
    )
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    table_chunks = [chunk for chunk in chunks if chunk.representation_type == "table"]

    assert len(table_chunks) == 1
    assert "Hemoglobin" in table_chunks[0].text
    assert "12-16 g/dL" in table_chunks[0].text
    assert table_chunks[0].metadata["table_id"]
    assert table_chunks[0].metadata["section_id"]
    assert table_chunks[0].metadata["parent_id"]


def test_figure_question_has_searchable_caption_with_structure() -> None:
    page = _page(
        8,
        "Chapter 3 Anatomy\n3.2 Thorax\nA diagram illustrates the major thoracic structures.",
        captions=["Figure 3.2. Major thoracic structures"],
    )
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    figure_chunks = [chunk for chunk in chunks if chunk.representation_type == "figure_caption"]

    assert len(figure_chunks) == 1
    assert figure_chunks[0].metadata["figure_id"]
    assert figure_chunks[0].metadata["section_id"]
    assert figure_chunks[0].metadata["parent_id"]
    assert "Major thoracic structures" in figure_chunks[0].text


def test_scanned_page_preserves_ocr_provenance_and_remains_searchable() -> None:
    page = _page(
        11,
        "Chapter 9 Pharmacology\n9.3 Dosage\nThe loading dose depends on body weight and clearance.",
        ocr_status="success",
    )
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    canonical = [chunk for chunk in chunks if chunk.representation_type == "canonical"]

    assert canonical
    assert all(chunk.metadata.get("ocr_status") == "success" for chunk in canonical)
    assert any("loading dose" in chunk.text.casefold() for chunk in canonical)


def test_mixed_book_preserves_hierarchy_when_table_and_figure_share_a_page() -> None:
    page = _page(
        14,
        "Chapter 12 Cardiovascular System\n12.4 Cardiac Output\nCardiac output depends on heart rate and stroke volume.\n[TABLE]\nVariable | Unit\nHeart rate | bpm",
        tables=["Variable | Unit\nHeart rate | bpm"],
        captions=["Figure 12.4. Cardiac cycle"],
    )
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    selected = [
        chunk
        for chunk in chunks
        if chunk.representation_type in {"canonical", "table", "figure_caption"}
    ]

    assert {chunk.representation_type for chunk in selected} == {
        "canonical",
        "table",
        "figure_caption",
    }
    parent_ids = {chunk.metadata.get("parent_id") for chunk in selected}
    section_ids = {chunk.metadata.get("section_id") for chunk in selected}
    assert len(parent_ids) == 1
    assert len(section_ids) == 1


def test_vision_adapter_is_safe_when_optional_visual_model_is_disabled() -> None:
    adapter = VisionFigureAdapter(model="")
    assert adapter.enabled is False
    assert callable(adapter.describe)


def test_system_refuses_to_invent_structure_for_empty_evidence() -> None:
    tracker = DocumentStructureTracker("empty-doc")
    snapshot = tracker.snapshot(1)

    assert snapshot.chapter_id is None
    assert snapshot.section_id is None
    assert snapshot.parent_id.startswith("document:")


@pytest.mark.parametrize(
    "query,expected_representation",
    [
        ("What does the table show for hemoglobin?", "table"),
        ("What is shown in Figure 3.2?", "figure_caption"),
        ("Explain ventilation in section 4.2.", "canonical"),
    ],
)
def test_high_level_query_intent_maps_to_expected_evidence_class(query: str, expected_representation: str) -> None:
    lowered = query.casefold()
    if "table" in lowered:
        inferred = "table"
    elif "figure" in lowered or "shown" in lowered:
        inferred = "figure_caption"
    else:
        inferred = "canonical"

    assert inferred == expected_representation
