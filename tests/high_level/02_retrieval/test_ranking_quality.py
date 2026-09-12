from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status, assert_rank_contains


@pytest.mark.high_level
def test_retrieval__exact_medical_term_ranks_in_top_three(clean_system):
    result = clean_system.answer("What is HbA1c used to assess?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    hits = result.get("hits") or []
    assert hits
    assert_rank_contains(hits, "HbA1c", top_k=3)


@pytest.mark.high_level
def test_retrieval__numeric_query_returns_number_and_unit_in_top_three(clean_system):
    result = clean_system.answer("What dose of metformin is stated?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    hits = result.get("hits") or []
    assert hits
    top_text = " ".join(str(getattr(hit, "text", "")) for hit in hits[:3])
    assert "500 mg" in top_text


@pytest.mark.high_level
def test_retrieval__table_query_prefers_table_evidence(clean_system):
    result = clean_system.answer("Which table contains the HbA1c target?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    hits = result.get("hits") or []
    assert hits
    assert "Table 1" in str(getattr(hits[0], "text", ""))
    assert float((hits[0].metadata or {}).get("intent_rerank_bonus", 0.0)) >= 0.40


@pytest.mark.high_level
def test_retrieval__simple_query_uses_at_most_two_variants_and_queries(clean_system):
    result = clean_system.answer("What is diabetes mellitus?")
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    route = result.get("route") or {}
    retrieval = result.get("retrieval") or {}
    variants = route.get("query_variants") or []
    assert len(variants) <= 2, variants
    assert int(retrieval.get("queries", 0)) <= 2, retrieval


@pytest.mark.high_level
def test_retrieval__document_filter_returns_only_target_document(clean_system, tmp_path):
    from tests.high_level.conftest import write_minimal_pdf

    extra = write_minimal_pdf(tmp_path / "other.pdf", ["OTHER_DOCUMENT_MARKER cardiology evidence."])
    ingestion = clean_system.ingest_file(extra)
    assert str(ingestion.get("status") or "").upper() == "READY"
    target_id = str(ingestion.get("document_id") or ingestion.get("id") or "")
    assert target_id

    result = clean_system.answer("What does OTHER_DOCUMENT_MARKER state?", metadata_filter={"document_id": target_id})
    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    hit_ids = {str(getattr(hit, "doc_id", "")) for hit in result.get("hits") or []}
    assert hit_ids == {target_id}