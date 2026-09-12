from __future__ import annotations

import re

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_status


@pytest.mark.high_level
def test_ingestion__persists_page_count_and_per_page_checkpoints(clean_system, tmp_path):
    pdf = write_minimal_pdf(tmp_path / "page_integrity.pdf", [
        "Page one clinical evidence marker.",
        "Page two clinical evidence marker.",
        "Page three clinical evidence marker.",
    ])
    result = clean_system.ingest_file(pdf)
    assert_status(result, {"READY"})

    document_id = str(result.get("document_id") or result.get("id") or "")
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert int(record.get("total_pages") or 0) == 3
    assert int(record.get("current_page") or 0) == 3

    with clean_system.state_store._connect() as connection:
        pages = connection.execute(
            "SELECT page_number, extraction_status, text FROM pages WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()
    assert [row[0] for row in pages] == [1, 2, 3]
    assert all(str(row[1]).upper() not in {"FAILED", "ERROR"} for row in pages)
    assert all(str(row[2] or "").strip() for row in pages)


@pytest.mark.high_level
def test_answer_citations__retain_source_identity_for_ready_document(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert_citations_valid(result)
    hits = result.get("hits") or []
    assert hits
    assert all(getattr(hit, "metadata", {}) for hit in hits)
    assert any("page" in str(getattr(hit, "metadata", {})).lower() or "page_numbers" in getattr(hit, "metadata", {}) for hit in hits)
    assert re.search(r"\[S\d+\]", str(result.get("answer") or ""))
