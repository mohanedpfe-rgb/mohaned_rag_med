from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_status


@pytest.mark.high_level
def test_security__path_traversal_upload_is_rejected_before_index_publication(clean_system, tmp_path):
    traversal = tmp_path / ".." / "outside.pdf"
    traversal.write_bytes(b"%PDF-1.4 invalid payload")

    with pytest.raises(ValueError, match="Unsupported or missing PDF|outside"):
        clean_system.ingest_file(traversal)


@pytest.mark.high_level
def test_security__non_pdf_upload_is_rejected_and_never_searchable(clean_system, tmp_path):
    payload = tmp_path / "notes.txt"
    payload.write_text("FAKE_MEDICAL_TEXT_THAT_MUST_NOT_BE_INDEXED", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported or missing PDF"):
        clean_system.ingest_file(payload)

    answer = clean_system.answer("What is FAKE_MEDICAL_TEXT_THAT_MUST_NOT_BE_INDEXED?")
    assert_exact_status(answer, "ABSTAIN")
    assert not answer.get("citations")
    assert not answer.get("hits")
