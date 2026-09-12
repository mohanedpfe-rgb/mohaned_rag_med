from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf


@pytest.mark.high_level
def test_storage__failed_input_does_not_leave_ready_document(clean_system, tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"invalid")
    result = clean_system.ingest_file(path)

    assert str(result.get("status") or "").upper() not in {"READY", "COMPLETED"}
    document_id = str(result.get("document_id") or result.get("id") or "")
    if document_id:
        record = clean_system.state_store.get_document(document_id)
        assert record is not None
        assert not clean_system.state_store.is_ready_status(record.get("status"))


@pytest.mark.high_level
def test_storage__metadata_filter_prevents_cross_document_language_leak(clean_system, tmp_path):
    english = write_minimal_pdf(tmp_path / "english_only.pdf", ["FILTER_LANGUAGE_EN: diabetes mellitus is a chronic metabolic disorder."])
    french = write_minimal_pdf(tmp_path / "french_only.pdf", ["FILTER_LANGUAGE_FR: le diabète est une maladie métabolique chronique."])
    assert str(clean_system.ingest_file(english).get("status") or "").upper() == "READY"
    assert str(clean_system.ingest_file(french).get("status") or "").upper() == "READY"

    result = clean_system.answer("What does FILTER_LANGUAGE_EN state?", {"language": "en"})
    assert result.get("hits")
    text = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
    assert "FILTER_LANGUAGE_EN" in text
    assert "FILTER_LANGUAGE_FR" not in text
