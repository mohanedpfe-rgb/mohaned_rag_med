import fitz
import pytest

from rag_project.ingestion.document_classifier import DocumentClassifier
from tests.high_level.conftest import write_minimal_pdf


@pytest.mark.high_level
def test_text_pdf__does_not_require_ocr_failure(clean_system, ready_document):
    result = clean_system.ingest_file(ready_document)
    assert isinstance(result, dict)
    assert str(result.get("status", "")).lower() not in {"ocr_required_but_disabled"}


@pytest.mark.high_level
def test_ocr_policy__is_explicit_in_runtime(clean_system):
    settings = clean_system.settings
    assert hasattr(settings, "ocr_enabled")
    assert hasattr(settings, "auto_ocr")
    assert hasattr(settings, "ocr_confidence_threshold")
    assert 0.0 <= float(settings.ocr_confidence_threshold) <= 1.0


@pytest.mark.high_level
def test_image_only_pdf_is_classified_as_scanned(scanned_document):
    result = DocumentClassifier.classify(scanned_document)
    assert result["document_type"] == "scanned_or_ocr_required"
    assert result["ocr_required"] is True
    assert result["text_pages"] == 0
    assert result["page_count"] == 2


@pytest.mark.high_level
def test_scanned_fixture_has_no_extractable_text(scanned_document):
    pdf = fitz.open(str(scanned_document))
    try:
        assert pdf.page_count == 2
        assert all(not page.get_text("text").strip() for page in pdf)
        assert all(page.get_images() for page in pdf)
    finally:
        pdf.close()


@pytest.mark.high_level
def test_large_and_empty_fixture_generators_are_valid(large_document, empty_document):
    large = fitz.open(str(large_document))
    empty = fitz.open(str(empty_document))
    try:
        assert large.page_count == 100
        assert empty.page_count == 1
        assert empty[0].get_text("text").strip() == ""
    finally:
        large.close()
        empty.close()
