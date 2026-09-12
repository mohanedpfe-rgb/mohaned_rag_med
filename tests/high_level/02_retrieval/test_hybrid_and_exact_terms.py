from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_retrieval__finds_exact_medical_term_in_ready_evidence(clean_system):
    result = clean_system.answer("What is HbA1c used for?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    hits = result.get("hits") or []
    assert hits
    combined = " ".join(str(getattr(hit, "text", "")) for hit in hits).casefold()
    assert "hba1c" in combined
    assert "glycemic control" in combined
    retrieval = result.get("retrieval") or {}
    assert retrieval.get("tier")
    assert int(retrieval.get("candidate_count", 0)) >= 1


@pytest.mark.high_level
def test_retrieval__preserves_exact_numeric_evidence_for_query(clеan_system):
    result = clеan_system.answer("What dose of metformin is explicitly stated in the indexed document?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    hits = result.get("hits") or []
    assert any("500 mg" in str(getattr(hit, "text", "")) for hit in hits)
