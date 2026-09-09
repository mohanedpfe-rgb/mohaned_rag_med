from __future__ import annotations

from rag_project.intelligence.query_intelligence import plan_query


def test_word_substring_does_not_trigger_relationship_intent():
    plan = plan_query("correlation of relatedness score")
    assert plan.intent != "relationship"


def test_numeric_terms_are_matched_as_terms():
    plan = plan_query("What is the mg value reported in the table?")
    assert plan.needs_numeric is True
    assert plan.needs_table is True


def test_comparison_term_is_not_triggered_by_unrelated_substring():
    plan = plan_query("difference between treatment A and treatment B")
    assert plan.intent == "comparison"


def test_normalization_and_variants_remain_bounded():
    plan = plan_query("x" * 10000)
    assert len(plan.normalized) <= 3000
    assert len(plan.variants) <= 8
