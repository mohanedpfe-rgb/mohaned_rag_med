from __future__ import annotations

import inspect
from pathlib import Path

import fitz

from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.configuration.config_i5_16gb import I5_16GB_PRESET
from rag_project.ingestion.document_models import PageExtraction
from rag_project.intelligence.deep_pdf_contract import install as install_deep
from rag_project.intelligence.deep_pdf_finalizer import VisionFigureAdapter
from rag_project.intelligence.deep_pdf_finalizer import install as install_final
from rag_project.intelligence.deep_pdf_finalizer_v2 import install as install_v2
from rag_project.intelligence.deep_pdf_finalizer_v3 import install as install_v3
from rag_project.intelligence.deep_pdf_finalizer_v4 import install as install_v4
from rag_project.intelligence.document_structure import (
    DocumentStructureStore,
    DocumentStructureTracker,
    STRUCTURE_SCHEMA_VERSION,
    extract_heading_candidates,
)
from rag_project.parsing.pdf_extractor import EXTRACTION_CACHE_VERSION, PDFExtractor
from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.storage.enhanced_vector_store import EnhancedVectorStore
from rag_project.storage.vector_store import VectorStore


install_deep()
install_final()
install_v2()
install_v3()
install_v4()


def _page(number: int = 1, text: str = "Chapter 1 Anatomy\n1.1 Bones\nText") -> PageExtraction:
    return PageExtraction(
        document_id="contract-doc",
        file_name="book.pdf",
        page_index=number - 1,
        page_number=number,
        text=text,
        extraction_method="pdf_text",
        quality_score=0.9,
        routing_decision="native",
        page_type="text_based",
    )


def test_all_45_pdf_architecture_contracts() -> None:
    checks: list[tuple[str, bool]] = []
    checks += [
        ("01_ocr_enabled", bool(I5_16GB_PRESET.get("ocr_enabled"))),
        ("02_cache_version", EXTRACTION_CACHE_VERSION >= "pdf-extractor-v3"),
        ("03_structure_schema", STRUCTURE_SCHEMA_VERSION >= 3),
        ("04_structure_store", inspect.isclass(DocumentStructureStore)),
        ("05_heading_candidates", bool(extract_heading_candidates("Chapter 2 Physiology\n2.1 Ventilation"))),
        ("06_tracker_cross_page", (lambda t: (lambda a, b: a.chapter_id == b.chapter_id and a.section_id == b.section_id and a.parent_id == b.parent_id)(t.analyze_page(1, "Chapter 2\n2.1 A"), t.analyze_page(2, "continued prose")))(DocumentStructureTracker("d"))),
        ("07_extractor_final_patch_hook", getattr(PDFExtractor, "_final_pdf_patched", False) and getattr(PDFExtractor, "_final_pdf_v2_patched", False) and getattr(PDFExtractor, "_final_pdf_v3_patched", False) and getattr(PDFExtractor, "_final_pdf_v4_patched", False)),
        ("08_chunker_final_patch_hook", getattr(SemanticChunker, "_final_pdf_patched", False) and getattr(SemanticChunker, "_final_pdf_v2_patched", False)),
        ("09_vector_final_patch_hook", getattr(VectorStore, "_final_pdf_patched", False) and getattr(VectorStore, "_final_pdf_v3_patched", False)),
        ("10_context_final_patch_hook", getattr(ContextBuilder, "_final_pdf_patched", False) and getattr(ContextBuilder, "_final_pdf_v2_patched", False)),
        ("11_hybrid_final_patch_hook", getattr(HybridRetriever, "_final_pdf_patched", False)),
        ("12_vision_adapter", inspect.isclass(VisionFigureAdapter)),
        ("13_book_hierarchy_store", inspect.isclass(EnhancedVectorStore)),
    ]

    page = _page()
    chunks = SemanticChunker(280, 20).chunk_pages([page])
    checks += [
        ("14_chunks_created", bool(chunks)),
        ("15_canonical_representation", any(c.representation_type == "canonical" for c in chunks)),
        ("16_parent_id", all(bool(c.metadata.get("parent_id")) for c in chunks if c.representation_type == "canonical")),
        ("17_section_id", all(bool(c.metadata.get("section_id")) for c in chunks if c.representation_type == "canonical")),
        ("18_chapter_id", all("chapter_id" in c.metadata for c in chunks if c.representation_type == "canonical")),
        ("19_hierarchy_path", all("hierarchy_path" in c.metadata for c in chunks if c.representation_type == "canonical")),
        ("20_page_numbers", all(c.metadata.get("page_numbers") for c in chunks)),
        ("21_quality", all(c.metadata.get("quality_score") is not None for c in chunks)),
        ("22_ocr_provenance", all("ocr_status" in c.metadata for c in chunks)),
    ]

    table_page = _page(2, "Chapter 1 Anatomy\n1.1 Bones\n[TABLE]\nAge | Male | Female\n20 | 1 | 2")
    table_page.table_texts = ["Age | Male | Female\n20 | 1 | 2"]
    table_page.table_ids = ["t1"]
    table_page.table_count = 1
    table_page.figure_captions = ["Figure 1. Bone diagram"]
    table_page.figure_ids = ["f1"]
    table_page.has_images = True
    table_page.image_count = 1
    specialized = SemanticChunker(280, 20).chunk_pages([table_page])
    checks += [
        ("23_table_unit", any(c.representation_type == "table" for c in specialized)),
        ("24_table_id", all(c.metadata.get("table_id") for c in specialized if c.representation_type == "table")),
        ("25_figure_caption_unit", any(c.representation_type == "figure_caption" for c in specialized)),
        ("26_figure_id", all(c.metadata.get("figure_id") for c in specialized if c.representation_type == "figure_caption")),
        ("27_specialized_parent_inherited", len({c.metadata.get("parent_id") for c in specialized if c.representation_type in {"canonical", "table", "figure_caption"}}) == 1),
        ("28_specialized_section_inherited", len({c.metadata.get("section_id") for c in specialized if c.representation_type in {"canonical", "table", "figure_caption"}}) == 1),
        ("29_visual_representation_constant", VisionFigureAdapter is not None),
    ]

    tracker = DocumentStructureTracker("cross")
    first = tracker.analyze_page(1, "Chapter 4 Respiratory System\n4.2 Ventilation")
    second = tracker.analyze_page(2, "continued prose")
    checks += [
        ("30_chapter_continuity", first.chapter_id == second.chapter_id),
        ("31_section_continuity", first.section_id == second.section_id),
        ("32_parent_continuity", first.parent_id == second.parent_id),
        ("33_stable_ids", first.section_id == tracker.snapshot(3).section_id),
    ]

    pdf = fitz.open()
    p = pdf.new_page(width=600, height=800)
    p.insert_text((50, 80), "CHAPTER 1", fontsize=24)
    p.insert_text((50, 120), "1.1 Anatomy", fontsize=16)
    p.insert_text((60, 170), "This is body text in the first column.", fontsize=10)
    p.insert_text((320, 170), "This is body text in the second column.", fontsize=10)
    pdf_path = Path("tests/data/_contract_layout.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(pdf_path)
    pdf.close()
    try:
        from rag_project.intelligence.deep_pdf_finalizer import _page_layout

        doc = fitz.open(pdf_path)
        layout = _page_layout(doc[0], "layout", 1)
        doc.close()
    finally:
        pdf_path.unlink(missing_ok=True)

    checks += [
        ("34_layout_schema", layout.get("layout_schema_version") == 1),
        ("35_layout_columns", int(layout.get("column_count") or 0) >= 1),
        ("36_layout_regions", bool(layout.get("text_regions"))),
        ("37_layout_reading_order", bool(layout.get("reading_order"))),
        ("38_layout_heading_candidates", bool(layout.get("heading_candidates"))),
        ("39_layout_table_regions_field", "table_regions" in layout),
        ("40_layout_figure_regions_field", "figure_regions" in layout),
    ]

    from rag_project.intelligence.deep_pdf_finalizer_v3 import _ocr_table_from_regions

    adapter = VisionFigureAdapter(model="")
    recovered = _ocr_table_from_regions(
        [
            {"bbox": [0.10, 0.10, 0.30, 0.14], "text": "Age"},
            {"bbox": [0.40, 0.10, 0.60, 0.14], "text": "Male"},
            {"bbox": [0.70, 0.10, 0.90, 0.14], "text": "Female"},
            {"bbox": [0.10, 0.18, 0.30, 0.22], "text": "20"},
            {"bbox": [0.40, 0.18, 0.60, 0.22], "text": "1"},
            {"bbox": [0.70, 0.18, 0.90, 0.22], "text": "2"},
        ]
    )
    checks += [
        ("41_vision_disabled_safely", adapter.enabled is False),
        ("42_vision_description_method", callable(adapter.describe)),
        ("43_ocr_geometry_table_recovery", bool(recovered and "Age | Male | Female" in recovered)),
        ("44_enhanced_store_search_method", callable(getattr(EnhancedVectorStore, "search", None))),
        ("45_runtime_contract_documented", "document hierarchy" in Path("ARCHITECTURE.md").read_text(encoding="utf-8").casefold()),
    ]

    failed = [name for name, ok in checks if not ok]
    assert len(checks) == 45
    assert not failed, "45-point PDF architecture contract failed: " + ", ".join(failed)
