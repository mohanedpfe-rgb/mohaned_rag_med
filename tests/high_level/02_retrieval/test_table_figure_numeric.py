from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_citations_valid, assert_exact_path, assert_exact_status, assert_grounded


@pytest.mark.high_level
def test_retrieval__table_and_figure_query_uses_exact_template_path_and_returns_markers(clean_system):
    result = clean_system.answer("What does Table 1 state about the HbA1c target, and what does Figure 1 represent?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    hits = result.get("hits") or []
    assert hits
    text = " ".join(str(getattr(hit, "text", "")) for hit in hits)
    assert "Table 1" in text
    assert "Figure 1" in text
    assert "7%" in text
    assert (result.get("route") or {}).get("intent") == "table"
    assert_citations_valid(result)
    assert_grounded(result)


@pytest.mark.high_level
def test_retrieval__numeric_query_marks_numeric_sensitivity_and_preserves_exact_value_on_template_path(clean_system):
    result = clean_system.answer("What numeric value is explicitly given for the HbA1c target?")

    assert_exact_status(result, "SUCCESS")
    assert_exact_path(result, "PATH_B_TEMPLATE")
    route = result.get("route") or {}
    assert route.get("numeric_sensitivity") is True
    evidence = result.get("evidence") or {}
    evidence_text = " ".join(str(value) for value in evidence.get("numeric_values", []))
    hit_text = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
    assert "7%" in evidence_text or "7%" in hit_text
    assert "7%" in str(result.get("answer") or "")
    assert_citations_valid(result)
    assert_grounded(result)
