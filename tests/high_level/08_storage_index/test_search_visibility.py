from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_storage__failed_document_never_becomes_searchable(clean_system, tmp_path, monkeypatch):
    pdf = write_minimal_pdf(tmp_path / "failed_visibility.pdf", [
        "UNSEARCHABLE_FAILED_DOCUMENT_MARKER 91c73b: must never be returned."
    ])

    monkeypatch.setattr(
        clean_system.vector_store,
        "validate_document_index",
        lambda *args, **kwargs: {"valid": False, "count": 0, "issues": ["forced failure"]},
    )
    ingestion = clean_system.ingest_file(pdf)
    assert str(ingestion.get("status") or "").upper() not in {"READY", "COMPLETED", "SUCCESS"}

    result = clean_system.answer("What is UNSEARCHABLE_FAILED_DOCUMENT_MARKER 91c73b?")
    status = str(result.get("status") or "").upper()
    assert status not in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    assert not (result.get("citations") or [])
    hit_text = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
    assert "UNSEARCHABLE_FAILED_DOCUMENT_MARKER" not in hit_text


@pytest.mark.high_level
def test_storage__ready_document_remains_searchable_after_an_unrelated_failure(clean_system, tmp_path, monkeypatch):
    before = clean_system.answer("What is diabetes mellitus?")
    assert_status(before, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})

    pdf = write_minimal_pdf(tmp_path / "unrelated_failed.pdf", [
        "UNRELATED_FAILED_ONLY_MARKER 4a11f: should not affect the healthy document."
    ])
    monkeypatch.setattr(
        clean_system.vector_store,
        "validate_document_index",
        lambda *args, **kwargs: {"valid": False, "count": 0, "issues": ["forced failure"]},
    )
    failed = clean_system.ingest_file(pdf)
    assert str(failed.get("status") or "").upper() not in {"READY", "COMPLETED", "SUCCESS"}

    after = clean_system.answer("What is diabetes mellitus?")
    assert_status(after, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    assert "UNRELATED_FAILED_ONLY_MARKER" not in str(after.get("answer") or "")
