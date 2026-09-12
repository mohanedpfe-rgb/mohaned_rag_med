from __future__ import annotations

import pytest

from tests.high_level.conftest import write_large_pdf
from tests.high_level.helpers import assert_exact_status


@pytest.mark.high_level
def test_ingestion__partial_embedding_failure_never_becomes_searchable(clean_system, tmp_path, monkeypatch):
    pdf = write_large_pdf(tmp_path / "partial_failure.pdf", pages=12)
    original_embed = clean_system.embedding_service.embed_texts
    calls = {"count": 0}

    def fail_mid_embedding(texts):
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("simulated mid-embedding failure")
        return original_embed(texts)

    monkeypatch.setattr(clean_system.embedding_service, "embed_texts", fail_mid_embedding)
    result = clean_system.ingest_file(pdf)

    assert_exact_status(result, "FAILED")
    assert calls["count"] >= 2
    document_id = str(result.get("document_id") or result.get("id") or "")
    assert document_id
    record = clean_system.state_store.get_document(document_id)
    assert record is not None
    assert str(record.get("status") or "").upper() == "FAILED_INDEXING"
    assert str(record.get("index_state") or "").upper() == "FAILED"
    assert clean_system.state_store.get_pages(document_id) == []

    answer = clean_system.answer("What does the controlled large-document page evidence marker state?")
    assert_exact_status(answer, "NOT_SUPPORTED")
    assert not answer.get("hits")
    assert not answer.get("citations")
