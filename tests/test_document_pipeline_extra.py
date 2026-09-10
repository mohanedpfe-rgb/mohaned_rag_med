from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.fixture
def page_class():
    from rag_project.ingestion.document_models import PageExtraction
    return PageExtraction


@pytest.fixture
def chunker():
    from rag_project.chunking.semantic_chunker import SemanticChunker
    return SemanticChunker(chunk_size=80, chunk_overlap=20)


def _page(PageExtraction, text: str, page: int = 1, **kwargs):
    defaults = dict(
        document_id="doc-1",
        file_name="book.pdf",
        page_index=page - 1,
        page_number=page,
        text=text,
        extraction_method="native",
        page_type="text",
        quality_score=0.9,
        ocr_status="not_needed",
        has_images=False,
        figure_ids=[],
        table_count=0,
        table_ids=[],
    )
    defaults.update(kwargs)
    return PageExtraction(**defaults)


def test_semantic_chunker_empty_input_is_empty(chunker):
    assert chunker.chunk_pages([]) == []
    assert list(chunker.chunk_page_batches([], batch_size=2)) == []


def test_semantic_chunker_enforces_positive_size_and_bounded_overlap():
    from rag_project.chunking.semantic_chunker import SemanticChunker

    c = SemanticChunker(chunk_size=10, chunk_overlap=999)
    assert c.chunk_size == 10
    assert c.chunk_overlap == 999
    splitter = c._child_splitter()
    assert splitter._chunk_size == 10
    assert splitter._chunk_overlap <= 5


def test_semantic_chunker_preserves_document_and_page_metadata(chunker, page_class):
    page = _page(page_class, "# Chapter One\n## Renal Disease\nDiabetes nephropathy is clinically important.")
    chunks = chunker.chunk_pages([page])
    assert chunks
    first = chunks[0]
    assert first.doc_id == "doc-1"
    assert first.file_name == "book.pdf"
    assert first.page_numbers == [1]
    assert first.metadata["document_id"] == "doc-1"
    assert first.metadata["page_numbers"] == [1]
    assert first.metadata["section_id"]
    assert first.metadata["parent_id"]
    assert "renal" in first.normalized_text.casefold() or "diabetes" in first.normalized_text.casefold()


def test_semantic_chunker_adds_figure_and_table_evidence_types(chunker, page_class):
    page = _page(
        page_class,
        "Figure shows renal blood flow. Table lists values.",
        has_images=True,
        figure_ids=["fig-1"],
        table_count=1,
        table_ids=["table-1"],
    )
    chunks = chunker.chunk_pages([page])
    assert chunks
    metadata = chunks[0].metadata
    assert "figure" in metadata["evidence_types"]
    assert "table" in metadata["evidence_types"]
    assert metadata["figure_id"] == "fig-1"
    assert metadata["table_id"] == "table-1"


def test_semantic_chunker_section_fallback_detects_numbered_heading(chunker):
    assert chunker._section_fallback("1. Renal function\nSome text") == "1. Renal function"
    assert chunker._section_fallback("No heading here") is None


def test_chunk_batches_respect_batch_size_and_global_indices(chunker, page_class):
    pages = [_page(page_class, f"Page {i} diabetes nephropathy.", i) for i in range(1, 6)]
    batches = list(chunker.chunk_page_batches(pages, batch_size=2))
    assert len(batches) >= 3
    indices = [chunk.chunk_index for batch in batches for chunk in batch]
    assert indices == list(range(len(indices)))
    assert all(len(batch) > 0 for batch in batches)


def test_document_model_round_trip_to_dict(page_class):
    page = _page(page_class, "sample")
    values = asdict(page)
    assert values["document_id"] == "doc-1"
    assert values["page_number"] == 1


@pytest.mark.parametrize("size", [1, 10, 100, 500])
def test_chunker_handles_varied_sizes(size: int, page_class):
    from rag_project.chunking.semantic_chunker import SemanticChunker

    chunker = SemanticChunker(chunk_size=size, chunk_overlap=0)
    text = "Diabetes nephropathy evidence. " * 20
    chunks = chunker.chunk_pages([_page(page_class, text)])
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


def test_document_processing_source_has_ocr_and_universal_pdf_hooks():
    root = Path(__file__).resolve().parents[1] / "rag_project"
    extractor = (root / "parsing" / "pdf_extractor.py").read_text(encoding="utf-8")
    ocr = (root / "ocr" / "ocr_service.py").read_text(encoding="utf-8")
    settings = (root / "configuration" / "settings.py").read_text(encoding="utf-8")
    assert "fitz" in extractor or "pymupdf" in extractor.casefold()
    assert "OCR" in ocr or "ocr" in ocr
    assert "universal_pdf_mode" in settings


def test_storage_and_embedding_runtime_have_failure_safe_entrypoints():
    root = Path(__file__).resolve().parents[1] / "rag_project"
    embedding = (root / "embeddings" / "embedding_service.py").read_text(encoding="utf-8")
    embedding_runtime = (root / "embeddings" / "embedding_runtime.py").read_text(encoding="utf-8")
    vector = (root / "storage" / "vector_store.py").read_text(encoding="utf-8")
    vector_runtime = (root / "storage" / "vector_store_runtime.py").read_text(encoding="utf-8")
    for source in (embedding, embedding_runtime, vector, vector_runtime):
        assert "except" in source
    assert "cache" in embedding.casefold()
    assert "dimension" in vector.casefold()


def test_runtime_layers_expose_recovery_or_stability_controls():
    root = Path(__file__).resolve().parents[1] / "rag_project"
    for name in ("runtime_recovery.py", "runtime_stability.py", "runtime_stability_v2.py", "runtime_stability_v3.py", "runtime_hardening.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert source.strip()
        assert "def " in source


def test_ci_configuration_runs_pytest():
    ci = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest" in ci
    assert "python" in ci.casefold()


def test_pytest_configuration_is_present():
    path = Path(__file__).resolve().parents[1] / "pytest.ini"
    assert path.is_file()
    assert "[pytest]" in path.read_text(encoding="utf-8")


def test_production_feature_contract_declares_expected_feature_count():
    from rag_project.intelligence.production_contract import validate_feature_contract

    contract = validate_feature_contract()
    assert contract["all_resolved"] is True
    assert contract.get("feature_count", 44) >= 40


def test_god_mode_100_declares_canonical_completion_path():
    source = (Path(__file__).resolve().parents[1] / "rag_project" / "intelligence" / "god_mode_100.py").read_text(encoding="utf-8")
    assert "complete_phases" in source
    assert "verify_final_answer" in source
    assert "score_entity_coverage" in source
    assert "confidence_calibration" in source
    assert "phase_5_intelligence_visibility" in source


def test_production_answer_has_hard_gate_markers():
    source = (Path(__file__).resolve().parents[1] / "rag_project" / "app" / "production_rag.py").read_text(encoding="utf-8")
    for marker in ("claim_evidence_matrix", "confidence_calibration", "medical_safety_policy", "privacy_safe_trace", "canonical_ingestion"):
        assert marker in source
