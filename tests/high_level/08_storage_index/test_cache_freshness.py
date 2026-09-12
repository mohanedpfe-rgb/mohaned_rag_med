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


@pytest.mark.high_level
def test_storage__metadata_filtered_query_never_reuses_global_semantic_cache(clean_system, tmp_path):
    target = write_minimal_pdf(tmp_path / "filter_target.pdf", ["FILTER_SCOPE_TARGET_MARKER: target endocrine evidence."])
    distractor = write_minimal_pdf(tmp_path / "filter_distractor.pdf", ["FILTER_SCOPE_DISTRACTOR_MARKER: distractor cardiology evidence."])
    target_result = clean_system.ingest_file(target)
    distractor_result = clean_system.ingest_file(distractor)
    assert str(target_result.get("status") or "").upper() == "READY"
    assert str(distractor_result.get("status") or "").upper() == "READY"

    target_id = str(target_result.get("document_id") or "")
    distractor_id = str(distractor_result.get("document_id") or "")
    assert target_id and distractor_id and target_id != distractor_id

    # Warm the global cache with the same query shape using the distractor evidence.
    warm = clean_system.answer("What does FILTER_SCOPE_TARGET_MARKER state?")
    assert_exact_status(warm, "SUCCESS")
    assert_exact_path(warm, "PATH_A_EXTRACTIVE")

    filtered = clean_system.answer(
        "What does FILTER_SCOPE_TARGET_MARKER state?",
        metadata_filter={"document_id": target_id},
    )
    assert_exact_status(filtered, "SUCCESS")
    assert_exact_path(filtered, "PATH_A_EXTRACTIVE")
    hit_ids = {str(getattr(hit, "doc_id", "")) for hit in filtered.get("hits") or []}
    assert hit_ids == {target_id}
    assert distractor_id not in hit_ids
    assert "FILTER_SCOPE_TARGET_MARKER" in str(filtered.get("answer") or "")
    assert_grounded(filtered)
    assert_citations_valid(filtered)