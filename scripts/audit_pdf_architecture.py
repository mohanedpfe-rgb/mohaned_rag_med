from __future__ import annotations

import inspect
from pathlib import Path

from rag_project.application import create_rag_system
from rag_project.chunking.semantic_chunker import SemanticChunker
from rag_project.configuration.config_i5_16gb import I5_16GB_PRESET
from rag_project.intelligence.document_structure import DocumentStructureStore, STRUCTURE_SCHEMA_VERSION
from rag_project.intelligence.deep_pdf_contract import install as install_deep_pdf
from rag_project.ingestion.document_models import PageExtraction
from rag_project.parsing.pdf_extractor import EXTRACTION_CACHE_VERSION, PDFExtractor
from rag_project.retrieval.context_builder import ContextBuilder
from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.storage.vector_store import VectorStore


def check(name: str, condition: bool) -> tuple[str, bool]:
    return name, bool(condition)


def main() -> int:
    install_deep_pdf()
    checks: list[tuple[str, bool]] = []
    checks.extend(
        [
            check("01 i5 OCR enabled", bool(I5_16GB_PRESET.get("ocr_enabled"))),
            check("02 extractor cache versioned", EXTRACTION_CACHE_VERSION >= "pdf-extractor-v3"),
            check("03 document structure schema >=3", STRUCTURE_SCHEMA_VERSION >= 3),
            check("04 durable structure store exists", inspect.isclass(DocumentStructureStore)),
            check("05 PDF extractor deep patch", bool(getattr(PDFExtractor, "_deep_pdf_patched", False))),
            check("06 semantic chunker global patch", bool(getattr(SemanticChunker, "_deep_structure_patched", False))),
            check("07 vector contract patch", bool(getattr(VectorStore, "_deep_contract_patched", False))),
            check("08 hierarchical retrieval patch", bool(getattr(HybridRetriever, "_deep_retrieval_patched", False))),
            check("09 context diversity patch", bool(getattr(ContextBuilder, "_deep_context_patched", False))),
            check("10 canonical source still present", Path("rag_project/ingestion/robust_ingestor.py").is_file()),
            check("11 runtime contract owns structure", "deep_pdf_contract" in Path("rag_project/runtime.py").read_text(encoding="utf-8")),
        ]
    )

    tracker = __import__("rag_project.intelligence.document_structure", fromlist=["DocumentStructureTracker"]).DocumentStructureTracker("audit")
    first = tracker.analyze_page(1, "Chapter 1 Anatomy\n1.1 Bones\nText")
    second = tracker.analyze_page(2, "More text")
    checks.extend(
        [
            check("12 global chapter survives page boundary", first.chapter_id == second.chapter_id),
            check("13 global section survives page boundary", first.section_id == second.section_id),
            check("14 global parent survives page boundary", first.parent_id == second.parent_id),
            check("15 chapter path exists", bool(first.hierarchy_path)),
            check("16 section id is stable", bool(first.section_id)),
        ]
    )

    page = PageExtraction(
        document_id="audit-doc",
        file_name="audit.pdf",
        page_index=0,
        page_number=1,
        text="Chapter 2 Physiology\n2.1 Ventilation\nText",
        extraction_method="pdf_text",
        quality_score=0.9,
        page_type="text_based",
    )
    chunks = SemanticChunker(300, 20).chunk_pages([page])
    metadata = chunks[0].metadata if chunks else {}
    checks.extend(
        [
            check("17 canonical chunks exist", bool(chunks)),
            check("18 canonical representation type", chunks and chunks[0].representation_type == "canonical"),
            check("19 canonical parent id", bool(metadata.get("parent_id"))),
            check("20 canonical section id", bool(metadata.get("section_id"))),
            check("21 canonical chapter id", bool(metadata.get("chapter_id"))),
            check("22 canonical hierarchy path", bool(metadata.get("hierarchy_path"))),
            check("23 canonical quality", metadata.get("quality_score") is not None),
        ]
    )

    system = create_rag_system()
    checks.extend(
        [
            check("24 runtime system builds", system is not None),
            check("25 structure collection hook", hasattr(system.vector_store, "_deep_structure_collection")),
            check("26 vector validation hook", hasattr(system.vector_store, "validate_document_index")),
            check("27 lexical storage present", Path(system.vector_store.lexical_database).exists()),
        ]
    )

    failed = [name for name, ok in checks if not ok]
    print(f"PDF_ARCHITECTURE_AUDIT {len(checks) - len(failed)}/{len(checks)} passed")
    for name, ok in checks:
        print(("PASS" if ok else "FAIL") + " " + name)
    if failed:
        print("FAILED_CHECKS:")
        for name in failed:
            print(" -", name)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
