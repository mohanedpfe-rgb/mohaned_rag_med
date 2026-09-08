from __future__ import annotations

from rag_project.intelligence.evidence_guard import (
    citation_firewall,
    grounding_decision,
    numeric_consistency,
    split_claims,
    verify_claims,
)
from rag_project.intelligence.god_mode import GOD_MODE_FEATURES, report
from rag_project.intelligence.pdf_intelligence import enrich_text, score_page_quality
from rag_project.intelligence.query_intelligence import plan_query


def test_exactly_44_features_are_declared_and_reported():
    assert len(GOD_MODE_FEATURES) == 44
    assert report()["feature_count"] == 44
    assert report()["fail_closed"] is True
    assert report()["universal_pdf_mode"] is True
    assert len(set(GOD_MODE_FEATURES)) == 44


def test_query_planner_covers_table_figure_comparison_and_multi_hop():
    plan = plan_query("Compare dose values in table 2 versus figure 3 and explain why they differ")
    assert plan.intent == "comparison"
    assert plan.needs_numeric
    assert plan.needs_table
    assert plan.needs_figure
    assert plan.needs_multi_hop
    assert plan.variants
    assert plan.subqueries


def test_pdf_intelligence_quality_and_enrichment_are_deterministic():
    text = "2. Treatment\nParacetamol 500 mg every 6 h. Success rate 80%."
    quality = score_page_quality(text, page_number=2, image_count=0)
    enriched = enrich_text(text)
    assert 0.0 <= quality.quality <= 1.0
    assert quality.route in {"native", "alternate_extractor", "ocr_verify", "ocr"}
    assert "500 mg" in enriched["normalized_text"]
    assert enriched["entities"]["drug"]
    assert enriched["headings"]


def test_numeric_consistency_rejects_unsupported_measurement():
    result = numeric_consistency("Dose is 600 mg.", "The recommended dose is 500 mg.")
    assert result["checked"] is True
    assert result["mismatch"] is True
    assert "600 mg" in result["unsupported_numeric"]


def test_claim_verification_detects_support_and_unsafe_numeric_claims():
    claims = verify_claims(
        "The dose is 500 mg. The dose is 600 mg.",
        ["The recommended dose is 500 mg."],
        ["S1"],
    )
    assert any(c.status == "SUPPORTED" for c in claims)
    assert any(c.status == "NUMERIC_MISMATCH" for c in claims)
    assert grounding_decision(claims)["allow"] is False


def test_citation_firewall_withholds_unsafe_claims():
    claims = verify_claims(
        "The dose is 600 mg.",
        ["The recommended dose is 500 mg."],
        ["S1"],
    )
    safe, used = citation_firewall("The dose is 600 mg.", claims)
    assert used is True
    assert "withheld" in safe.lower() or "verified" in safe.lower()


def test_claim_splitter_keeps_multi_sentence_answer_manageable():
    claims = split_claims("The patient received treatment today. The patient improved after treatment. The evidence reports a measured response.")
    assert len(claims) == 3
