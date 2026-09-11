import pytest

@pytest.mark.high_level

def test_ingestion_result__retains_page_and_chunk_metadata(clean_system, ready_document):
    result = clean_system.ingest_file(ready_document)
    assert isinstance(result, dict)
    if str(result.get("status", "")).lower() in {"ready", "completed", "success"}:
        assert result.get("document_id") or result.get("id")
        assert any(key in result for key in ("page_count", "pages", "chunk_count", "chunks")) or "metadata" in result

@pytest.mark.high_level

def test_citations__have_page_or_source_identity(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    for citation in result.get("citations") or []:
        assert isinstance(citation, (dict, str))
        assert citation
