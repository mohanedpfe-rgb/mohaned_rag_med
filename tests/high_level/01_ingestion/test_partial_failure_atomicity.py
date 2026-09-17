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
    # The system has other indexed documents (ready_document), so it may return results
    # from those. The key is that the failed document itself should not be searchable.
    # Since the ready_document contains similar markers, we accept either status.
    assert str(answer.get("status") or "").upper() in {"NOT_SUPPORTED", "GENERATION_ABSTAIN", "SUCCESS"}
    # If we got a success, verify it's not from the failed document
    if str(answer.get("status") or "").upper() == "SUCCESS":
        hits = answer.get("hits") or []
        for hit in hits:
            hit_doc_id = str((hit.metadata or {}).get("document_id", hit.doc_id))
            assert hit_doc_id != document_id, f"Failed document {document_id} should not be searchable"
