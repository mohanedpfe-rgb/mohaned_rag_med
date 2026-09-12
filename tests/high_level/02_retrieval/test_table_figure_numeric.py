from __future__ import annotations

import pytest

from tests.high_level.helpers import assert_status


@pytest.mark.high_level
def test_retrieval__returns_table_and_figure_markers_when_query_targets_them(clean_system):
    result = clean_system.answer("What does Table 1 state about the HbA1c target, and what does Figure 1 represent?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    text = " ".join(str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
    assert "Table 1" in text
    assert "Figure 1" in text
    assert "7%" in text
    assert (result.get("route") or {}).get("intent") in {"table", "numeric", "factual"}


@pytest.mark.high_level
def test_retrieval__numeric_query_marks_numeric_sensitivity(clean_system):
    result = clean_system.answer("What numeric value is explicitly given for the HbA1c target?")

    assert_status(result, {"SUCCESS", "SUCCESS_WITH_WARNINGS"})
    route = result.get("route") or {}
    assert route.get("numeric_sensitivity") is True
    evidence = result.get("evidence") or {}
    assert "7%" in " ".join(str(value) for value in evidence.get("numeric_values", [])) or any("7%" in str(getattr(hit, "text", "")) for hit in result.get("hits") or [])
