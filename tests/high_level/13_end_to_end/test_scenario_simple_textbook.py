import pytest
from tests.high_level.conftest import write_minimal_pdf

@pytest.mark.high_level

def test_simple_textbook_scenario__upload_then_answer(clean_system, tmp_path):
    path = write_minimal_pdf(tmp_path / "clean_diabetes_en.pdf", ["Diabetes mellitus is a chronic metabolic disorder. HbA1c assesses glycemic control."])
    ingestion = clean_system.ingest_file(path)
    result = clean_system.answer("What is diabetes mellitus?")
    assert ingestion.get("status")
    assert result.get("status")
    assert result.get("pipeline_authority")
