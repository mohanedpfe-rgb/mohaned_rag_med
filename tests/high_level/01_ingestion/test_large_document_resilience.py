import pytest
from tests.high_level.conftest import write_minimal_pdf

@pytest.mark.high_level
@pytest.mark.slow

def test_large_document__ends_in_terminal_state(clean_system, tmp_path):
    source = write_minimal_pdf(tmp_path / "large_100pages.pdf", [f"Page {i}: diabetes mellitus and HbA1c." for i in range(1, 101)])
    result = clean_system.ingest_file(source)
    assert str(result.get("status", "")).lower() in {"ready", "completed", "success", "failed", "skipped"}

@pytest.mark.high_level

def test_ingestion_settings__bound_worker_and_batch_sizes(clean_system):
    settings = clean_system.settings
    assert 1 <= settings.max_workers <= 4
    assert 1 <= settings.embedding_batch_size <= 32
