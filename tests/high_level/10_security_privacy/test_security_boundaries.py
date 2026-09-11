import pytest
from pathlib import Path

@pytest.mark.high_level

def test_non_pdf_upload__is_rejected_or_failed(clean_system, tmp_path):
    source = tmp_path / "not_pdf.txt"
    source.write_text("patient data", encoding="utf-8")
    result = clean_system.ingest_file(source)
    assert str(result.get("status", "")).lower() not in {"ready", "success"}

@pytest.mark.high_level

def test_runtime_paths__are_project_scoped(settings_i5):
    for path in (settings_i5.incoming_dir, settings_i5.processed_dir, settings_i5.failed_dir, settings_i5.vector_db_dir):
        assert Path(path).resolve().is_relative_to(settings_i5.project_root.resolve())

@pytest.mark.high_level

def test_default_ollama_url__is_local(settings_i5):
    assert settings_i5.ollama_base_url.startswith("http://127.0.0.1:") or settings_i5.ollama_base_url.startswith("http://localhost:")
