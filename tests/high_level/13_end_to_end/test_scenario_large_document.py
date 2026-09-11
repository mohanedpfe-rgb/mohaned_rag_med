import pytest
from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_latency_under

@pytest.mark.high_level
@pytest.mark.slow

def test_large_document_scenario__ingests_and_answers_with_ceiling(clean_system, tmp_path):
    path = write_minimal_pdf(tmp_path / "large_100pages.pdf", [f"Page {i}: Diabetes mellitus is a chronic metabolic disorder. HbA1c assesses glycemic control." for i in range(1, 101)])
    ingestion = clean_system.ingest_file(path)
    assert ingestion.get("status")
    result = clean_system.answer("What is diabetes mellitus?")
    assert result.get("status")
    assert_latency_under(result, max(15.0, float(clean_system.settings.generation_latency_budget_seconds) + 2.0))
