import pytest
from tests.high_level.conftest import write_minimal_pdf

@pytest.mark.high_level

def test_scanned_mixed_scenario__terminates_cleanly(clean_system, tmp_path):
    path = write_minimal_pdf(tmp_path / "scanned_mixed.pdf", ["Scanned-source surrogate: diabetes mellitus.", "Numeric statement: HbA1c value 7.0%."] * 4)
    ingestion = clean_system.ingest_file(path)
    assert str(ingestion.get("status") or "").lower() in {"ready", "completed", "success", "failed", "skipped"}
