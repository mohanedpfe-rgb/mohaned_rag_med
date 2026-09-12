from __future__ import annotations

import pytest

from tests.high_level.conftest import write_large_pdf
from tests.high_level.helpers import assert_exact_status


@pytest.mark.high_level
@pytest.mark.slow
def test_ingestion__hundred_page_pdf_reaches_ready_without_partial_state(clean_system, tmp_path):
    document = write_large_pdf(tmp_path / "100_pages.pdf", pages=100)
    result = clean_system.ingest_file(document)

    assert_exact_status(result, "READY")
    document_id = str(result.get("document_id") or result.get("id") or "")
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert clean_system.state_store.is_ready_status(record.get("status"))
    assert str(record.get("index_state") or "").upper() == "READY"
    assert int(record.get("total_pages") or 0) == 100
    assert int(record.get("current_page") or 0) == 100
    assert not record.get("error")


@pytest.mark.high_level
@pytest.mark.slow
def test_ingestion__large_document_records_ready_terminal_process_event(clean_system, tmp_path):
    document = write_large_pdf(tmp_path / "events_100_pages.pdf", pages=100)
    result = clean_system.ingest_file(document)
    assert_exact_status(result, "READY")

    document_id = str(result.get("document_id") or result.get("id") or "")
    events = clean_system.state_store.get_events(document_id)
    assert events
    assert any(str(event.get("stage") or "").upper() in {"INDEXING", "VALIDATING_INDEX", "READY"} for event in events)
    assert str(events[-1].get("status") or "").upper() == "READY"
