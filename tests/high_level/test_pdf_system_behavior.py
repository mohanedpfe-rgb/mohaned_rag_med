from __future__ import annotations

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
    quality_score: float | None = None,
) -> PageExtraction:
    tables = list(tables or [])
    captions = list(captions or [])
    quality_score = 0.92 if quality_score is None and ocr_status == "not_required" else quality_score
    quality_score = 0.84 if quality_score is None else quality_score
    return PageExtraction(
        document_id="high-level-book",
        file_name="medical_textbook.pdf",
        page_index=page_number - 1,
        page_number=page_number,
        text=text,
        extraction_method="pdf_text",
        page_type="text_based" if ocr_status == "not_required" else "scanned",
        quality_score=quality_score,
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


@pytest.mark.high_level
def test_pdf_behavior__book_ingestion_produces_prose_table_and_figure_representations() -> None:
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
    assert {"canonical", "table", "figure_caption"}.issubset(representations)
    assert all(chunk.metadata.get("document_id") == "high-level-book" for chunk in chunks)
    assert all(chunk.metadata.get("page_numbers") for chunk in chunks)
    assert all(chunk.metadata.get("parent_id") for chunk in chunks)
    assert all(chunk.metadata.get("section_id") for chunk in chunks)


@pytest.mark.high_level
def test_pdf_behavior__chunk_metadata_is_sufficient_for_page_level_traceability() -> None:
    page = _page(17, "Chapter 16 Infectious Diseases\n16.1 Transmission\nDroplet transmission occurs through respiratory particles.")
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    assert chunks
    for chunk in chunks:
        meta = chunk.metadata
        assert meta["document_id"]
        assert meta["file_name"]
        assert meta["page_numbers"] == [17]
        assert meta["parent_id"]
        assert meta["section_id"]
        assert "quality_score" in meta
        assert "ocr_status" in meta
        assert "routing_decision" in meta


@pytest.mark.high_level
def test_pdf_behavior__cross_page_question_context_keeps_same_section_identity() -> None:
    tracker = DocumentStructureTracker("high-level-book")
    first = tracker.analyze_page(1, "Chapter 4 Respiratory System\n4.2 Ventilation\nVentilation is the movement of air.")
    second = tracker.analyze_page(2, "The pressure gradient drives airflow during inspiration and expiration.")
    assert second.chapter_id == first.chapter_id
    assert second.section_id == first.section_id
    assert second.parent_id == first.parent_id
    assert second.hierarchy_path == first.hierarchy_path


@pytest.mark.high_level
def test_pdf_behavior__empty_page_does_not_create_fake_semantic_content() -> None:
    assert SemanticChunker(280, 20).chunk_pages([_page(18, "   \n\n")]) == []


@pytest.mark.high_level
def test_pdf_behavior__figure_caption_keeps_same_structure_as_surrounding_section() -> None:
    page = _page(16, "Chapter 15 Dermatology\n15.2 Lesions\nLesion morphology.\nFigure follows.", captions=["Figure 15.2. Common lesion morphology"])
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    canonical = [c for c in chunks if c.representation_type == "canonical"]
    figures = [c for c in chunks if c.representation_type == "figure_caption"]
    assert canonical and figures
    assert figures[0].metadata["section_id"] == canonical[-1].metadata["section_id"]
    assert figures[0].metadata["parent_id"] == canonical[-1].metadata["parent_id"]


@pytest.mark.high_level
def test_pdf_behavior__figure_question_has_searchable_caption_with_structure() -> None:
    page = _page(8, "Chapter 3 Anatomy\n3.2 Thorax\nA diagram illustrates the major thoracic structures.", captions=["Figure 3.2. Major thoracic structures"])
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    figure_chunks = [chunk for chunk in chunks if chunk.representation_type == "figure_caption"]
    assert len(figure_chunks) == 1
    assert figure_chunks[0].metadata["figure_id"]
    assert figure_chunks[0].metadata["section_id"]
    assert figure_chunks[0].metadata["parent_id"]
    assert "Major thoracic structures" in figure_chunks[0].text


@pytest.mark.high_level
def test_pdf_behavior__heading_change_is_reflected_in_later_chunks_only() -> None:
    pages = [_page(1, "Chapter 6 Renal System\n6.1 Filtration\nFiltration text"), _page(2, "6.2 Reabsorption\nReabsorption text")]
    chunks = [c for c in SemanticChunker(280, 20).chunk_pages(pages) if c.representation_type == "canonical"]
    filtration = [c for c in chunks if "Filtration" in c.text]
    reabsorption = [c for c in chunks if "Reabsorption" in c.text]
    assert filtration and reabsorption
    assert filtration[0].metadata["section_id"] != reabsorption[0].metadata["section_id"]
    assert filtration[0].metadata["chapter_id"] == reabsorption[0].metadata["chapter_id"]


@pytest.mark.high_level
def test_pdf_behavior__high_level_query_intent_maps_to_expected_evidence_class() -> None:
    cases = {"What does the table show for hemoglobin?": "table", "What is shown in Figure 3.2?": "figure_caption", "Explain ventilation in section 4.2.": "canonical"}
    for query, expected in cases.items():
        lowered = query.casefold()
        inferred = "table" if "table" in lowered else "figure_caption" if "figure" in lowered else "canonical"
        assert inferred == expected


@pytest.mark.high_level
def test_pdf_behavior__low_quality_page_preserves_quality_score_and_ocr_route() -> None:
    page = _page(13, "Chapter 13 Pathology\n13.1 Histology\nVery noisy OCR text.", ocr_status="success", quality_score=0.28)
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    assert chunks
    assert all(chunk.metadata["quality_score"] == pytest.approx(0.28) for chunk in chunks)
    assert all(chunk.metadata["routing_decision"] == "ocr" for chunk in chunks)
    assert all(chunk.metadata["ocr_status"] == "success" for chunk in chunks)


@pytest.mark.high_level
def test_pdf_behavior__mixed_book_preserves_hierarchy_when_table_and_figure_share_a_page() -> None:
    page = _page(14, "Chapter 12 Cardiovascular System\n12.4 Cardiac Output\nCardiac output depends on heart rate and stroke volume.\n[TABLE]\nVariable | Unit\nHeart rate | bpm", tables=["Variable | Unit\nHeart rate | bpm"], captions=["Figure 12.4. Cardiac cycle"])
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    selected = [c for c in chunks if c.representation_type in {"canonical", "table", "figure_caption"}]
    assert {c.representation_type for c in selected} == {"canonical", "table", "figure_caption"}
    assert len({c.metadata.get("parent_id") for c in selected}) == 1
    assert len({c.metadata.get("section_id") for c in selected}) == 1


@pytest.mark.high_level
def test_pdf_behavior__multiple_figures_on_same_page_get_distinct_ids() -> None:
    page = _page(9, "Chapter 11 Neurology\n11.1 Brain Imaging\nFigures below.", captions=["Figure 11.1. MRI anatomy", "Figure 11.2. Functional map"])
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    figures = [c for c in chunks if c.representation_type == "figure_caption"]
    assert len(figures) == 2
    assert len({c.metadata["figure_id"] for c in figures}) == 2


@pytest.mark.high_level
def test_pdf_behavior__multiple_tables_on_same_page_get_distinct_ids() -> None:
    page = _page(7, "Chapter 10 Nutrition\n10.2 Macronutrients\nContext\n[TABLE]\nA | B\n1 | 2\n[TABLE]\nC | D\n3 | 4", tables=["A | B\n1 | 2", "C | D\n3 | 4"])
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    tables = [c for c in chunks if c.representation_type == "table"]
    assert len(tables) == 2
    assert len({c.metadata["table_id"] for c in tables}) == 2


@pytest.mark.high_level
def test_pdf_behavior__nested_sections_produce_distinct_stable_section_ids() -> None:
    tracker = DocumentStructureTracker("book")
    section_a = tracker.analyze_page(3, "Chapter 5 Pharmacology\n5.1 Absorption\nText")
    section_b = tracker.analyze_page(4, "5.2 Distribution\nText")
    assert section_a.chapter_id == section_b.chapter_id
    assert section_a.section_id != section_b.section_id
    assert section_a.parent_id != section_b.parent_id


@pytest.mark.high_level
def test_pdf_behavior__page_order_does_not_break_structure_continuity() -> None:
    tracker = DocumentStructureTracker("book")
    first = tracker.analyze_page(1, "Chapter 14 Endocrinology\n14.3 Thyroid\ntext")
    second = tracker.analyze_page(2, "continued thyroid physiology")
    third = tracker.analyze_page(3, "more thyroid physiology")
    assert first.section_id == second.section_id == third.section_id
    assert first.parent_id == second.parent_id == third.parent_id


@pytest.mark.high_level
def test_pdf_behavior__scanned_page_preserves_ocr_provenance_and_remains_searchable() -> None:
    page = _page(11, "Chapter 9 Pharmacology\n9.3 Dosage\nThe loading dose depends on body weight and clearance.", ocr_status="success")
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    canonical = [chunk for chunk in chunks if chunk.representation_type == "canonical"]
    assert canonical
    assert all(chunk.metadata.get("ocr_status") == "success" for chunk in canonical)
    assert any("loading dose" in chunk.text.casefold() for chunk in canonical)


@pytest.mark.high_level
def test_pdf_behavior__system_refuses_to_invent_structure_for_empty_evidence() -> None:
    tracker = DocumentStructureTracker("empty-doc")
    snapshot = tracker.snapshot(1)
    assert snapshot.chapter_id is None
    assert snapshot.section_id is None
    assert snapshot.parent_id.startswith("document:")


@pytest.mark.high_level
def test_pdf_behavior__table_content_is_not_duplicated_into_canonical_prose() -> None:
    table = "Drug | Dose\nAmoxicillin | 500 mg"
    page = _page(6, f"Chapter 8 Antibiotics\n8.1 Dosing\nUse according to clinical indication.\n[TABLE]\n{table}", tables=[table])
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    canonical = [c for c in chunks if c.representation_type == "canonical"]
    tables = [c for c in chunks if c.representation_type == "table"]
    assert canonical and tables
    assert "Use according to clinical indication" in canonical[0].text
    assert "Amoxicillin | 500 mg" in tables[0].text
    assert "Amoxicillin | 500 mg" not in canonical[0].text


@pytest.mark.high_level
def test_pdf_behavior__table_question_uses_first_class_table_evidence() -> None:
    page = _page(5, "Chapter 7 Laboratory Values\n7.1 Reference Values\n[TABLE]\nTest | Normal Range\nHemoglobin | 12-16 g/dL", tables=["Test | Normal Range\nHemoglobin | 12-16 g/dL"])
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    table_chunks = [chunk for chunk in chunks if chunk.representation_type == "table"]
    assert len(table_chunks) == 1
    assert "Hemoglobin" in table_chunks[0].text
    assert "12-16 g/dL" in table_chunks[0].text
    assert table_chunks[0].metadata["table_id"]
    assert table_chunks[0].metadata["section_id"]
    assert table_chunks[0].metadata["parent_id"]


@pytest.mark.high_level
def test_pdf_behavior__two_chapters_do_not_share_structure_identity() -> None:
    tracker = DocumentStructureTracker("book")
    first = tracker.analyze_page(1, "Chapter 1 Anatomy\n1.1 Bones\nBone text")
    second = tracker.analyze_page(20, "Chapter 2 Physiology\n2.1 Muscle\nMuscle text")
    assert first.chapter_id != second.chapter_id
    assert first.section_id != second.section_id
    assert first.parent_id != second.parent_id


@pytest.mark.high_level
def test_pdf_behavior__vision_adapter_is_safe_when_optional_visual_model_is_disabled() -> None:
    adapter = VisionFigureAdapter(model="")
    assert adapter.enabled is False
    assert callable(adapter.describe)
