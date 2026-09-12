from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_security__path_traversal_upload_is_rejected_before_index_publication(clean_system, tmp_path):
    traversal = tmp_path / ".." / "outside.pdf"
    traversal.write_bytes(b"not relevant")

    result = clean_system.ingest_file(traversal)
    status = str(result.get("status") or "").upper()
    assert status not in {"READY", "COMPLETED", "SUCCESS"}
    assert not result.get("citations")


@pytest.mark.high_level
def test_security__non_pdf_upload_is_rejected_and_never_searchable(clean_system, tmp_path):
    payload = tmp_path / "notes.txt"
    payload.write_text("FAKE_MEDICAL_TEXT_THAT_MUST_NOT_BE_INDEXED", encoding="utf-8")

    result = clean_system.ingest_file(payload)
    status = str(result.get("status") or "").upper()
    assert status not in {"READY", "COMPLETED", "SUCCESS"}

    answer = clean_system.answer("What is FAKE_MEDICAL_TEXT_THAT_MUST_NOT_BE_INDEXED?")
    assert_status(answer, {"NOT_SUPPORTED", "GENERATION_ABSTAIN", "ANSWER_UNAVAILABLE", "BLOCK"})
    assert not answer.get("citations")
