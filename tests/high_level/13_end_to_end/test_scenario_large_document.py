from __future__ import annotations

import pytest

from tests.high_level.conftest import write_large_pdf
from tests.high_level.helpers import assert_citations_valid, assert_document_ready, assert_exact_path, assert_exact_status, assert_grounded, assert_latency_under


@pytest.mark.high_level
@pytest.mark.slow
def test_e2e_large_document__hundred_page_pdf_reaches_ready_and_answers_page_100_fast(clean_system, tmp_path):
    document = write_large_pdf(tmp_path / "large_100_page.pdf", pages=100)
    ingestion = clean_system.ingest_file(document)
    assert str(ingestion.get("status") or "").upper() == "READY"

    document_id = str(ingestion.get("document_id") or ingestion.get("id") or "")
    record = assert_document_ready(clean_system, document_id)
    assert int(record.get("total_pages") or 0) == 100

    result = clean_system.answer("What clinical evidence marker appears on controlled large-document page 100?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert_latency_under(result, 5.0)
    hits = result.get("hits") or []
    assert hits
    combined = " ".join(str(getattr(hit, "text", "")) for hit in hits)
    assert "page 100" in combined.casefold()
    assert all(str(getattr(hit, "doc_id", "")) == document_id for hit in hits)
    assert_citations_valid(result)
    assert_grounded(result)
