from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.runtime_stability_v3 import _guard_transition


def test_state_transition_rejects_ready_regression(tmp_path):
    store = IngestionStateStore(tmp_path / "state.db")
    store.upsert_document(
        {
            "document_id": "doc-1",
            "content_hash": "hash-1",
            "file_path": str(tmp_path / "a.pdf"),
            "file_name": "a.pdf",
            "file_size": 1,
            "current_stage": "READY",
            "status": "READY",
            "parser_version": "test",
        }
    )

    class GuardHarness:
        def get_document(self, document_id):
            return store.get_document(document_id)

        def _runtime_v3_original_transition(self, *args, **kwargs):
            return None

    harness = GuardHarness()
    with pytest.raises(RuntimeError, match="Invalid state regression"):
        _guard_transition(harness, "doc-1", "EXTRACTING", current_page=0, total_pages=1)


def test_metric_enrichment_reads_persisted_json():
    document = {
        "document_id": "doc-1",
        "ingestion_metrics": json.dumps({"chunk_count": 7, "embedding_count": 7, "page_count": 3}),
    }
    from rag_project.runtime_stability_v3 import _merge_metrics

    result = _merge_metrics(dict(document))
    assert result["chunk_count"] == 7
    assert result["embedding_count"] == 7
    assert result["page_count"] == 3


def test_health_snapshot_does_not_require_embedding_probe():
    from rag_project.runtime_stability_v3 import _health_report_fast

    fake = SimpleNamespace(
        embedding_service=SimpleNamespace(_ollama_available=None, last_error="offline", dimension=None, identity=None),
        vector_store=SimpleNamespace(
            compatibility_report=lambda identity: {"status": "UNKNOWN"},
            count=lambda: 0,
        ),
        settings=SimpleNamespace(embedding_model="embed", generation_model="gen"),
        _production_feature_contract={"all_resolved": True},
    )
    report = _health_report_fast(fake)
    assert report["ready"] is False
    assert report["pipeline"]["non_blocking_health"] is True
