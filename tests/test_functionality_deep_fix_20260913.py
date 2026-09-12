from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.evidence_guard import verify_claims
from rag_project.intelligence.med_evidence_pro import AnswerCascade, EvidenceClaim, RouteMetadata
from rag_project.runtime_functionality_deep_fix import (
    _citation_complete_without_shared_state,
    _filter_false_numeric_contradictions,
    _functionality_sentences,
    _wrap_generate,
    _wrap_numeric_verifier,
    _wrap_retrieval_cache_fallthrough,
    _wrap_retrieval_skip_health_probe,
    _wrap_route,
)


def _route() -> RouteMetadata:
    return RouteMetadata(
        intent="factual", complexity=0.90, entities=("diabetes",), numeric_sensitivity=False,
        temporal_sensitivity=False, conditional_context=(), is_follow_up=False, confidence_threshold=0.75,
        template_type=None, retrieval_timeout_ms=2000, needs_multi_hop=True, query_variants=("diabetes",),
    )


class _Cache:
    def __init__(self): self.deleted: list[str] = []
    def delete(self, query: str) -> None: self.deleted.append(query)


def test_irrelevant_cache_hit_falls_through_to_fresh_retrieval():
    cache = _Cache(); calls = {"count": 0}
    retriever = SimpleNamespace(cache=cache, system=SimpleNamespace())
    def original(self, question, route, where=None):
        calls["count"] += 1
        return ([], {"tier": "CACHE", "cache_hit": True}) if calls["count"] == 1 else (["fresh-hit"], {"tier": "TIER1", "cache_hit": False})
    hits, state = _wrap_retrieval_cache_fallthrough(original)(retriever, "diabetes", _route())
    assert hits == ["fresh-hit"] and state["tier"] == "TIER1" and calls["count"] == 2 and cache.deleted == ["diabetes"]


def test_health_probe_is_not_allowed_to_consume_real_retrieval_call():
    calls: list[tuple[str, int, object]] = []
    class Store:
        def retrieve(self, query, top_k=8, where=None): calls.append((query, top_k, where)); return ["real-hit"]
    retriever = SimpleNamespace(system=SimpleNamespace(retriever=Store()))
    def original(self, question, route, where=None):
        self.system.retriever.retrieve(question, 1, where)
        return self.system.retriever.retrieve(question, 8, where), {"tier": "TIER1"}
    hits, state = _wrap_retrieval_skip_health_probe(original)(retriever, "diabetes", _route(), {"doc": "1"})
    assert hits == ["real-hit"] and state["tier"] == "TIER1" and calls == [("diabetes", 8, {"doc": "1"})]


def test_llm_citation_validation_does_not_depend_on_shared_system_hit_state():
    route = _route(); compiled = {"claims": [EvidenceClaim("Diabetes is a metabolic disorder.", (2,), False, False, 0.95)], "compressed": "- Diabetes is a metabolic disorder. [S2]"}
    system = SimpleNamespace(_med_selected_hits=[]); cascade = AnswerCascade(system)
    original_citation = AnswerCascade._citation_complete
    AnswerCascade._citation_complete = staticmethod(_citation_complete_without_shared_state(original_citation))
    try:
        original_generate = AnswerCascade.generate; AnswerCascade.generate = _wrap_generate(original_generate)
        cascade._llm = lambda question, evidence, route: "Diabetes is a metabolic disorder. [S2]"
        answer, path, _ = cascade.generate("What is diabetes?", route, compiled)
    finally:
        AnswerCascade._citation_complete = original_citation
        AnswerCascade.generate = original_generate
    assert answer.endswith("[S2]") and path == "PATH_C_CONSTRAINED_LLM"


def test_citation_guard_uses_current_generation_limit_not_shared_state():
    original = AnswerCascade._citation_complete; guarded = _citation_complete_without_shared_state(original)
    import rag_project.runtime_functionality_deep_fix as fix
    previous = getattr(fix._TLS, "citation_limit", None); fix._TLS.citation_limit = 2
    try: assert guarded("A grounded medical statement. [S2]", 0) is True
    finally:
        if previous is None: del fix._TLS.citation_limit
        else: fix._TLS.citation_limit = previous


def test_citation_guard_rejects_nonexistent_evidence_id_within_range():
    import rag_project.runtime_functionality_deep_fix as fix
    guarded = _citation_complete_without_shared_state(AnswerCascade._citation_complete)
    previous_ids = getattr(fix._TLS, "citation_ids", None); previous_limit = getattr(fix._TLS, "citation_limit", None)
    fix._TLS.citation_ids = {2}; fix._TLS.citation_limit = 5
    try: assert guarded("A grounded medical statement. [S1]", 5) is False
    finally:
        if previous_ids is None: del fix._TLS.citation_ids
        else: fix._TLS.citation_ids = previous_ids
        if previous_limit is None: del fix._TLS.citation_limit
        else: fix._TLS.citation_limit = previous_limit


def test_followup_guard_does_not_misclassify_compound_question():
    def original(self, question, context, safety): return replace_route(_route(), is_follow_up=True)
    route = _wrap_route(original)(SimpleNamespace(), "What causes cirrhosis and how is it managed?", "", SimpleNamespace())
    assert route.is_follow_up is False


def test_followup_guard_preserves_explicit_followup():
    def original(self, question, context, safety): return replace_route(_route(), is_follow_up=False)
    route = _wrap_route(original)(SimpleNamespace(), "And what about treatment?", "previous context", SimpleNamespace())
    assert route.is_follow_up is True


def replace_route(route: RouteMetadata, **changes) -> RouteMetadata:
    from dataclasses import replace
    return replace(route, **changes)


def test_short_evidence_sentences_are_not_discarded():
    assert _functionality_sentences("DKA.\nNo.") == ["DKA.", "No."]


def test_equivalent_units_are_not_reported_as_contradiction():
    original_result = {"has_contradiction": True, "conflicts": [{"left": ["1 g"], "right": ["1000 mg"]}], "agreement": 0.65}
    result = _filter_false_numeric_contradictions(lambda claims: original_result)([])
    assert result["has_contradiction"] is False and result["conflicts"] == [] and result["agreement"] == 1.0


def test_real_numeric_contradiction_is_preserved():
    original_result = {"has_contradiction": True, "conflicts": [{"left": ["500 mg"], "right": ["1000 mg"]}], "agreement": 0.65}
    result = _filter_false_numeric_contradictions(lambda claims: original_result)([])
    assert result["has_contradiction"] is True and result["conflicts"] and result["agreement"] == 0.65


def test_numeric_verifier_does_not_clear_mismatch_from_only_one_valid_value():
    hit = SimpleNamespace(text="The dose is 1 g; the range elsewhere is 500 mg.", metadata={})
    route = SimpleNamespace(numeric_sensitivity=True)
    original_result = {"numeric_mismatch": True, "grounding": {"allow": True}, "final_answer": {"allow": True}, "allow": False}
    wrapped = _wrap_numeric_verifier(lambda self, answer, hits, route, compiled: original_result)
    result = wrapped(SimpleNamespace(), "Use 1000 mg and 750 mg.", [hit], route, {})
    assert result["numeric_mismatch"] is True and result["allow"] is False


def test_numeric_verifier_clears_mismatch_when_all_values_are_equivalent():
    hit = SimpleNamespace(text="The dose is 1 g and 500 mg.", metadata={})
    route = SimpleNamespace(numeric_sensitivity=True)
    original_result = {"numeric_mismatch": True, "grounding": {"allow": True}, "final_answer": {"allow": True}, "allow": False}
    wrapped = _wrap_numeric_verifier(lambda self, answer, hits, route, compiled: original_result)
    result = wrapped(SimpleNamespace(), "Use 1000 mg and 0.5 g.", [hit], route, {})
    assert result["numeric_mismatch"] is False and result["allow"] is True


def test_numeric_claim_must_match_the_cited_source_not_another_source():
    answer = "The dose is 500 mg. [S1]"
    evidence = ["The dose is 250 mg.", "The dose is 500 mg."]
    checks = verify_claims(answer, evidence, ["S1", "S2"])
    assert len(checks) == 1
    assert checks[0].status == "NUMERIC_MISMATCH"
    assert checks[0].numeric_mismatch is True


def test_numeric_claim_passes_when_the_cited_source_contains_an_equivalent_unit():
    answer = "The dose is 1000 mg. [S1]"
    evidence = ["The dose is 1 g.", "The dose is 500 mg."]
    checks = verify_claims(answer, evidence, ["S1", "S2"])
    assert len(checks) == 1
    assert checks[0].status == "SUPPORTED"
    assert checks[0].numeric_mismatch is False
