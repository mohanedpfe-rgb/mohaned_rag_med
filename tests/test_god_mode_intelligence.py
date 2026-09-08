from __future__ import annotations

from rag_project.intelligence.evidence_guard import citation_firewall, numeric_consistency, verify_claims
from rag_project.intelligence.god_mode import GOD_MODE_FEATURES, report
from rag_project.intelligence.pdf_intelligence import enrich_text, score_page_quality
from rag_project.intelligence.query_intelligence import decompose_query, plan_query


def test_exactly_44_features_are_declared() -> None:
    assert len(GOD_MODE_FEATURES) == 44
    assert report()["feature_count"] == 44
    assert report()["fail_closed"] is True


def test_page_quality_routes_scanned_page_to_ocr() -> None:
    quality = score_page_quality("", page_number=7, image_count=1)
    assert quality.route == "ocr"
    assert "likely_scanned" in quality.warnings


def test_numeric_normalization_and_entities() -> None:
    enriched = enrich_text("Metformin 500 mg twice daily; 7.5% target")
    assert "500 mg" in enriched["normalized_text"]
    assert enriched["entities"]["dose"]
    assert enriched["entities"]["percent"] == ["7.5%"]


def test_query_planner_decomposes_numeric_question() -> None:
    pieces = decompose_query("dose and contraindications of metformin")
    plan = plan_query("What is the dose and contraindications of metformin?")
    assert len(pieces) >= 2
    assert plan.intent in {"numeric", "multi_part"}
    assert plan.needs_table is True
    assert plan.needs_multi_hop is True
    assert plan.variants


def test_numeric_mismatch_is_detected() -> None:
    result = numeric_consistency("Use 100 mg twice daily", "The recommended dose is 10 mg once daily")
    assert result["checked"] is True
    assert result["mismatch"] is True


def test_claim_verifier_fails_closed_for_unsupported_numbers() -> None:
    claims = verify_claims(
        "Use 100 mg twice daily. The drug is recommended.",
        ["The recommended dose is 10 mg once daily. The drug is discussed."],
        ["S1"],
    )
    assert any(c.status == "NUMERIC_MISMATCH" for c in claims)
    safe, firewall = citation_firewall("Use 100 mg twice daily.", claims)
    assert firewall is True
    assert "withheld" in safe.lower()
