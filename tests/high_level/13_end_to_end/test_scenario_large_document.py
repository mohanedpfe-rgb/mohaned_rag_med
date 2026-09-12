from __future__ import annotations

import pytest

from tests.high_level.conftest import write_large_pdf
from tests.high_level.helpers import assert_status


@pytest.mark.high_level
@pytest.mark.slow
def test_e2e_large_document__hundred_page_pdf_reaches_ready_and_remains_queryable(clean_system, tmp_path):
    document = write_large_pdf(tmp_path / "large_100_page.pdf", pages=100)
    ingestion = clean_system.ingest_file(document)
    assert_status(ingestion, {"READY"})

    document_id = str(ingestion.get("document_id") or ingestion.get("id") or "")
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert int(record.get("total_pages") or 0) == 100
    assert clean_system.state_store.is_ready_status(record.get("status"))
    assert str(record.get("index_state") or "").upper() == "READY"

    result = clean_system.answer("What clinical evidence marker appears on controlled large-document page 100?")
    assert str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS", "NOT_SUPPORTED", "GENERATION_ABSTAIN"}
    if result.get("hits"):
        combined = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
        assert "page 100" in combined.casefold() or "page 99" in combined.casefold()
