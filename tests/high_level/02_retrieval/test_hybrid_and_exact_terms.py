from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_exact_path, assert_exact_status, assert_rank_contains


@pytest.mark.high_level
def test_retrieval__finds_exact_medical_term_in_top_three_ready_evidence(clean_system):
    result = clean_system.answer("What is HbA1c used for?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_A_EXTRACTIVE")
    hits = result.get("hits") or []
    assert hits
    assert_rank_contains(hits, "HbA1c", top_k=3)
    combined = " ".join(str(getattr(hit, "text", "")) for hit in hits[:3]).casefold()
    assert "glycemic control" in combined
    retrieval = result.get("retrieval") or {}
    assert retrieval.get("tier")
    assert int(retrieval.get("candidate_count", 0)) >= 1


@pytest.mark.high_level
def test_retrieval__preserves_exact_numeric_evidence_in_top_three_hits(clean_system):
    result = clean_system.answer("What dose of metformin is explicitly stated in the indexed document?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    hits = result.get("hits") or []
    assert hits
    assert_rank_contains(hits, "500 mg", top_k=3)
