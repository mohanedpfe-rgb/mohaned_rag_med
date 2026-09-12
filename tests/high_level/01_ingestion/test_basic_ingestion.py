from __future__ import annotations

import hashlib

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_document_ready, assert_status


@pytest.mark.high_level
def test_ingestion__ready_requires_full_page_completion_and_exact_content_hash(clean_system, tmp_path):
    pdf = write_minimal_pdf(tmp_path / "unique_ingestion.pdf", [
        "Diabetes mellitus is a chronic metabolic disorder.",
        "HbA1c is used to assess glycemic control.",
    ])

    result = clean_system.ingest_file(pdf)
    assert str(result.get("status") or "").upper() == "READY"
    document_id = str(result.get("document_id") or result.get("id") or "")
    assert_document_ready(clean_system, document_id)

    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert clean_system.state_store.is_ready_status(record["status"])
    assert str(record.get("index_state", "")).upper() == "READY"
    assert int(record.get("total_pages") or 0) == 2
    assert int(record.get("current_page") or 0) == int(record.get("total_pages") or 0)
    assert record.get("error") in (None, "")
    assert str(record.get("content_hash")) == hashlib.sha256(pdf.read_bytes()).hexdigest()


@pytest.mark.high_level
def test_ingest_directory__two_valid_pdfs_each_publish_one_ready_terminal_document(clean_system, temp_project_root):
    incoming = temp_project_root / "data" / "incoming"
    write_minimal_pdf(incoming / "a_unique.pdf", ["Hypertension is persistent high blood pressure."])
    write_minimal_pdf(incoming / "b_unique.pdf", ["Anemia is a reduction in red blood cell mass."])

    results = clean_system.ingest_directory(incoming)

    assert len(results) == 2
    assert [str(item.get("status") or "").upper() for item in results] == ["READY", "READY"]
    ready_ids = [str(item.get("document_id") or item.get("id") or "") for item in results]
    assert all(ready_ids)
    assert len(ready_ids) == len(set(ready_ids))
