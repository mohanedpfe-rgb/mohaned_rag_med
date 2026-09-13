from __future__ import annotations

from types import SimpleNamespace

from rag_project.runtime_functionality_final_audit_fix import (
    _percent_equivalent,
    _wrap_followup_route,
    _wrap_percent_contradictions,
    _wrap_percent_numeric_verifier,
)
from rag_project.intelligence.med_evidence_pro import RouteMetadata


def _route() -> RouteMetadata:
    return RouteMetadata(
        intent="factual",
        complexity=0.4,
        entities=("diabetes",),
        numeric_sensitivity=False,
        temporal_sensitivity=False,
        conditional_context=(),
        is_follow_up=True,
        confidence_threshold=0.75,
        template_type=None,
        retrieval_timeout_ms=1200,
        needs_multi_hop=False,
        query_variants=("diabetes",),
    )


def test_compound_question_is_not_marked_follow_up_by_internal_and():
    route = _wrap_followup_route(lambda self, question, context, safety: _route())(
        SimpleNamespace(),
        "What causes cirrhosis and how is it managed?",
        "",
        SimpleNamespace(),
    )
    assert route.is_follow_up is False


def test_leading_and_question_remains_follow_up():
    route = _wrap_followup_route(lambda self, question, context, safety: _route())(
        SimpleNamespace(),
        "And what about treatment?",
        "previous context",
        SimpleNamespace(),
    )
    assert route.is_follow_up is True


def test_percent_values_are_parsed_and_equivalent():
    assert _percent_equivalent("5%", "5%") is True
    assert _percent_equivalent("5.0 %", "5%") is True
    assert _percent_equivalent("5%", "6%") is False


def test_percent_numeric_verifier_accepts_grounded_percentage():
    hit = SimpleNamespace(text="The risk is 5%.", metadata={})
    route = SimpleNamespace(numeric_sensitivity=True)
    original_result = {
        "numeric_mismatch": True,
        "grounding": {"allow": True},
        "final_answer": {"allow": True},
        "allow": False,
    }
    wrapped = _wrap_percent_numeric_verifier(lambda self, answer, hits, route, compiled: original_result)
    result = wrapped(SimpleNamespace(), "The risk is 5%.", [hit], route, {})
    assert result["numeric_mismatch"] is False
    assert result["allow"] is True


def test_percent_numeric_verifier_rejects_ungrounded_percentage():
    hit = SimpleNamespace(text="The risk is 5%.", metadata={})
    route = SimpleNamespace(numeric_sensitivity=True)
    original_result = {
        "numeric_mismatch": False,
        "grounding": {"allow": True},
        "final_answer": {"allow": True},
        "allow": True,
    }
    wrapped = _wrap_percent_numeric_verifier(lambda self, answer, hits, route, compiled: original_result)
    result = wrapped(SimpleNamespace(), "The risk is 6%.", [hit], route, {})
    assert result["numeric_mismatch"] is True
    assert result["allow"] is False


def test_equivalent_percentage_contradiction_is_removed():
    result = _wrap_percent_contradictions(
        lambda claims: {
            "has_contradiction": True,
            "conflicts": [{"left": ["5%"], "right": ["5.0%"]}],
            "agreement": 0.65,
        }
    )([])
    assert result["has_contradiction"] is False
    assert result["conflicts"] == []
    assert result["agreement"] == 1.0
