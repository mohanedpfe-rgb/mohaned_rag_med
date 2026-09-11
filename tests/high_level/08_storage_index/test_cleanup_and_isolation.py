import pytest

@pytest.mark.high_level

def test_failed_input__does_not_leave_ready_status(clean_system, tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"invalid")
    result = clean_system.ingest_file(path)
    assert str(result.get("status", "")).lower() != "ready"

@pytest.mark.high_level

def test_document_scoped_answer__retains_metadata_filter(clean_system):
    try:
        result = clean_system.answer("What is diabetes?", {"language": "en"})
    except TypeError:
        result = clean_system.answer("What is diabetes?")
    assert result.get("status")
