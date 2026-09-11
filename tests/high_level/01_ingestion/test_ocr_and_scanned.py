import pytest
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
