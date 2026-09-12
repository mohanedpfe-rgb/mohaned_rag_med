from __future__ import annotations

import pytest

from rag_project.security import validate_storage_path
from tests.high_level.helpers import assert_exact_status


@pytest.mark.high_level
def test_security__path_traversal_storage_configuration_is_rejected():
    root = __import__("pathlib").Path("/tmp/bookrag-security-root")
    outside = root / ".." / "outside" / "data"
    with pytest.raises(ValueError, match="must remain inside"):
        validate_storage_path(root, outside, "incoming_dir")


@pytest.mark.high_level
def test_security__non_pdf_upload_is_rejected_and_never_searchable(clean_system, tmp_path):
    payload = tmp_path / "notes.txt"
    payload.write_text("FAKE_MEDICAL_TEXT_THAT_MUST_NOT_BE_INDEXED", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported or missing PDF"):
        clean_system.ingest_file(payload)

    answer = clean_system.answer("What is FAKE_MEDICAL_TEXT_THAT_MUST_NOT_BE_INDEXED?")
    assert_exact_status(answer, "NOT_SUPPORTED")
    assert not answer.get("generation_path")
    assert not answer.get("citations")
    assert not answer.get("hits")
