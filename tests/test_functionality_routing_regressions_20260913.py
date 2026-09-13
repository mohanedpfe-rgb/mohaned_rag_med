from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.med_evidence_pro import QueryRouter
from rag_project.runtime_functionality_routing_fix import _wrap_route


def test_numeric_comparison_keeps_comparison_intent_and_template():
    router = QueryRouter()
    safety = SimpleNamespace(confidence_threshold=0.75)
    route = _wrap_route(lambda self, question, context, safety: SimpleNamespace(
        intent="numeric",
        numeric_sensitivity=True,
        template_type="dosage",
    ))(router, "Compare the doses of metformin and insulin", "", safety)
    assert route.intent == "comparison"
    assert route.template_type == "comparison"


def test_non_comparison_numeric_query_stays_numeric():
    router = QueryRouter()
    safety = SimpleNamespace(confidence_threshold=0.75)
    route = _wrap_route(lambda self, question, context, safety: SimpleNamespace(
        intent="numeric",
        numeric_sensitivity=True,
        template_type="dosage",
    ))(router, "What is the dose of metformin?", "", safety)
    assert route.intent == "numeric"
    assert route.template_type == "dosage"
