from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_grounded, assert_status


@pytest.mark.high_level
def test_retrieval__returns_table_and_figure_markers_when_query_targets_them(clean_system):
    result = clean_system.answer("What does Table 1 state about the HbA1c target, and what does Figure 1 represent?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    hits = result.get("hits") or []
    assert hits
    text = " ".join(str(getattr(hit, "text", "")) for hit in hits)
    assert "Table 1" in text
    assert "Figure 1" in text
    assert "7%" in text
    assert (result.get("route") or {}).get("intent") in {"table", "numeric", "factual"}
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_retrieval__numeric_query_marks_numeric_sensitivity_and_preserves_exact_value(clean_system):
    result = clean_system.answer("What numeric value is explicitly given for the HbA1c target?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    route = result.get("route") or {}
    assert route.get("numeric_sensitivity") is True
    evidence = result.get("evidence") or {}
    evidence_text = " ".join(str(value) for value in evidence.get("numeric_values", []))
    hit_text = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
    assert "7%" in evidence_text or "7%" in hit_text
    assert "7%" in str(result.get("answer") or "")
    assert_citations_valid(result)
    assert_grounded(result)
