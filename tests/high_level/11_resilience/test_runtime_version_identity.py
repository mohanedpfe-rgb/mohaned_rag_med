from types import SimpleNamespace

import pytest

from rag_project.intelligence.runtime_safety import ready_hits
from rag_project.retrieval.hybrid_retriever import RetrievalHit


class _FakeStateStore:
    def __init__(self, record):
        self.record = record

    def get_document(self, document_id):
        return self.record if document_id == self.record["document_id"] else None

    @staticmethod
    def is_ready_status(status):
        return str(status or "").upper() in {"READY", "COMPLETED", "SUCCESS"}


@pytest.mark.high_level
def test_ready_hits_accepts_legacy_content_hash_version_without_weakening_ready_gate():
    record = {
        "document_id": "doc-1",
        "status": "READY",
        "index_state": "READY",
        "version_id": "ingestion-version-1",
        "content_hash": "content-hash-1",
    }
    system = SimpleNamespace(state_store=_FakeStateStore(record))
    hit = RetrievalHit(
        "doc-1",
        "Diabetes mellitus is a chronic metabolic disorder.",
        {
            "document_id": "doc-1",
            "index_state": "READY",
            # Legacy rows incorrectly stored content_hash in version_id.
            "version_id": "content-hash-1",
        },
        0.9,
        0.9,
        0.9,
    )

    assert ready_hits(system, [hit]) == [hit]


@pytest.mark.high_level
def test_ready_hits_rejects_unknown_version_even_when_legacy_mode_is_enabled():
    record = {
        "document_id": "doc-1",
        "status": "READY",
        "index_state": "READY",
        "version_id": "ingestion-version-1",
        "content_hash": "content-hash-1",
    }
    system = SimpleNamespace(state_store=_FakeStateStore(record))
    hit = RetrievalHit(
        "doc-1",
        "Diabetes mellitus is a chronic metabolic disorder.",
        {
            "document_id": "doc-1",
            "index_state": "READY",
            "version_id": "different-document-version",
        },
        0.9,
        0.9,
        0.9,
    )

    assert ready_hits(system, [hit]) == []
