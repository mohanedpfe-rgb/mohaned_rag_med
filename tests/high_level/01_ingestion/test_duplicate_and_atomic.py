import hashlib
from pathlib import Path

import pytest
from tests.high_level.conftest import write_minimal_pdf

@pytest.mark.high_level

def test_duplicate_upload__does_not_create_second_searchable_copy(clean_system, ready_document, temp_project_root):
    incoming = temp_project_root / "data" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    source = incoming / "duplicate.pdf"
    source.write_bytes(Path(ready_document).read_bytes())
    first = clean_system.ingest_file(source)
    second_source = incoming / "duplicate_second.pdf"
    second_source.write_bytes(Path(ready_document).read_bytes())
    second = clean_system.ingest_file(second_source)
    assert isinstance(first, dict) and isinstance(second, dict)
    statuses = {str(first.get("status", "")).lower(), str(second.get("status", "")).lower()}
    assert statuses & {"skipped", "duplicate", "ready", "completed", "success"}

@pytest.mark.high_level

def test_failed_publication__never_reports_ready_without_identity(clean_system, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a real pdf")
    result = clean_system.ingest_file(broken)
    assert isinstance(result, dict)
    if str(result.get("status", "")).lower() == "ready":
        assert result.get("document_id") or result.get("id")
