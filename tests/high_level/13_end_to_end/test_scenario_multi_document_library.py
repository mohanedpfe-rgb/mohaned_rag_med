from __future__ import annotations

import pytest

from tests.high_level.conftest import write_minimal_pdf


@pytest.mark.high_level
def test_e2e_multi_document__library_search_returns_only_relevant_source(clean_system, tmp_path):
    source_a = write_minimal_pdf(tmp_path / "cardiology.pdf", [
        "DOC_CARDIO: myocardial infarction is a myocardial injury caused by ischemia.",
    ])
    source_b = write_minimal_pdf(tmp_path / "endocrine.pdf", [
        "DOC_ENDO: diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia.",
    ])
    result_a = clean_system.ingest_file(source_a)
    result_b = clean_system.ingest_file(source_b)
    assert str(result_a.get("status") or "").upper() == "READY"
    assert str(result_b.get("status") or "").upper() == "READY"

    result = clean_system.answer("What does DOC_ENDO state about diabetes mellitus?")
    assert str(result.get("status") or "").upper() in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
    hits = result.get("hits") or []
    assert hits
    text = " ".join(str(getattr(hit, "text", "")) for hit in hits)
    assert "DOC_ENDO" in text
    assert "DOC_CARDIO" not in text
