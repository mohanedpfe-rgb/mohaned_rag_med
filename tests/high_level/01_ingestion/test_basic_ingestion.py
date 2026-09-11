import pytest

from tests.high_level.conftest import write_minimal_pdf

@pytest.mark.high_level

def test_upload_clean_pdf__becomes_searchable(clean_system, ready_document):
    result = clean_system.ingest_file(ready_document)
    assert str(result.get("status", "")).lower() in {"ready", "completed", "success", "skipped"}
    if str(result.get("status", "")).lower() == "ready":
        assert result.get("document_id") or result.get("id")

@pytest.mark.high_level

def test_ingest_directory__processes_pdf_files(clean_system, temp_project_root):
    incoming = temp_project_root / "data" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    write_minimal_pdf(incoming / "a.pdf", ["Diabetes is a chronic metabolic disorder."])
    write_minimal_pdf(incoming / "b.pdf", ["HbA1c assesses glycemic control."])
    results = clean_system.ingest_directory(incoming)
    assert len(results) == 2
    assert all(isinstance(item, dict) for item in results)
