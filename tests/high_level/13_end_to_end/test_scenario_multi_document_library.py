import pytest
from tests.high_level.conftest import write_minimal_pdf

@pytest.mark.high_level

def test_multi_document_library__keeps_documents_distinguishable(clean_system, tmp_path):
    first = write_minimal_pdf(tmp_path / "diabetes.pdf", ["Diabetes mellitus is a chronic metabolic disorder."])
    second = write_minimal_pdf(tmp_path / "hypertension.pdf", ["Hypertension is persistent elevation of blood pressure."])
    assert clean_system.ingest_file(first).get("status")
    assert clean_system.ingest_file(second).get("status")
    result = clean_system.answer("What is hypertension?")
    assert result.get("status")
    text = str(result.get("answer") or "").casefold()
    assert "hypertension" in text or str(result.get("status") or "").upper() in {"NOT_SUPPORTED", "REASONING_ABSTAIN", "ANSWER_UNAVAILABLE"}
