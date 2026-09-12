from __future__ import annotations

import pytest

from rag_project.intelligence.med_evidence_pro import RetrievalHit, SemanticCache
from rag_project.intelligence.runtime_safety import cached_result_is_fresh, invalidate_stale_cache
from tests.high_level.conftest import write_minimal_pdf
from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_storage__stale_cached_evidence_is_invalidated_after_document_leaves_ready(clean_system, tmp_path):
    pdf = write_minimal_pdf(tmp_path / "cache_stale.pdf", ["CACHE_STALE_MARKER is controlled evidence."])
    ingestion = clean_system.ingest_file(pdf)
    assert str(ingestion.get("status") or "").upper() == "READY"
    document_id = str(ingestion.get("document_id") or ingestion.get("id") or "")
    record = clean_system.state_store.get_document(document_id)
    assert record is not None

    cache = SemanticCache(clean_system.settings.project_root / "data" / "med_evidence_cache.sqlite3")
    cached_hit = RetrievalHit(
        document_id,
        "CACHE_STALE_MARKER is controlled evidence.",
        {"document_id": document_id, "version_id": record.get("version_id"), "index_state": "READY", "chunk_id": "cache-stale-1", "page_numbers": [1]},
        0.99,
    )
    cache.put("What is CACHE_STALE_MARKER?", [cached_hit])
    assert cached_result_is_fresh(clean_system, [cached_hit]) is True

    clean_system.state_store.update_document(document_id, status="FAILED_INDEXING", index_state="FAILED")
    assert cached_result_is_fresh(clean_system, [cached_hit]) is False
    assert invalidate_stale_cache(clean_system, "What is CACHE_STALE_MARKER?") is True
    assert cache.get("What is CACHE_STALE_MARKER?") is None


@pytest.mark.high_level
def test_storage__cache_rebuild_cannot_resurrect_non_ready_evidence(clean_system, tmp_path):
    pdf = write_minimal_pdf(tmp_path / "cache_rebuild.pdf", ["CACHE_REBUILD_MARKER is ready evidence."])
    ingestion = clean_system.ingest_file(pdf)
    assert str(ingestion.get("status") or "").upper() == "READY"
    result = clean_system.answer("What is CACHE_REBUILD_MARKER?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    assert_citations_valid(result)
    assert_grounded(result)
    hits = result.get("hits") or []
    assert hits
    assert all(str((hit.metadata or {}).get("index_state", "")).upper() == "READY" for hit in hits)
