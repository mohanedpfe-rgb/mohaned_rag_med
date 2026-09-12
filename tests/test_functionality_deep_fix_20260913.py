from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.med_evidence_pro import AnswerCascade, EvidenceClaim, RouteMetadata
from rag_project.runtime_functionality_deep_fix import (
    _citation_complete_without_shared_state,
    _filter_false_numeric_contradictions,
    _functionality_sentences,
    _wrap_generate,
    _wrap_retrieval_cache_fallthrough,
    _wrap_retrieval_skip_health_probe,
    _wrap_route,
)


def _route() -> RouteMetadata:
    return RouteMetadata(
        intent="factual",
        complexity=0.90,
        entities=("diabetes",),
        numeric_sensitivity=False,
        temporal_sensitivity=False,
        conditional_context=(),
        is_follow_up=False,
        confidence_threshold=0.75,
        template_type=None,
        retrieval_timeout_ms=2000,
        needs_multi_hop=True,
        query_variants=("diabetes",),
    )


class _Cache:
    def __init__(self):
        self.deleted: list[str] = []

    def delete(self, query: str) -> None:
        self.deleted.append(query)


def test_irrelevant_cache_hit_falls_through_to_fresh_retrieval():
    cache = _Cache()
    calls = {"count": 0}

    class Retriever:
        pass

    retriever = Retriever()
    retriever.cache = cache
    retriever.system = SimpleNamespace()

    def original(self, question, route, where=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return [], {"tier": "CACHE", "cache_hit": True}
        return ["fresh-hit"], {"tier": "TIER1", "cache_hit": False}

    wrapped = _wrap_retrieval_cache_fallthrough(original)
    hits, state = wrapped(retriever, "diabetes", _route())

    assert hits == ["fresh-hit"]
    assert state["tier"] == "TIER1"
    assert calls["count"] == 2
    assert cache.deleted == ["diabetes"]


def test_health_probe_is_not_allowed_to_consume_real_retrieval_call():
    calls: list[tuple[str, int, object]] = []

    class Store:
        def retrieve(self, query, top_k=8, where=None):
            calls.append((query, top_k, where))
            return ["real-hit"]

    class Retriever:
        pass

    retriever = Retriever()
    retriever.system = SimpleNamespace(retriever=Store())

    def original(self, question, route, where=None):
        self.system.retriever.retrieve(question, 1, where)
        hits = self.system.retriever.retrieve(question, 8, where)
        return hits, {"tier": "TIER1"}

    wrapped = _wrap_retrieval_skip_health_probe(original)
    hits, state = wrapped(retriever, "diabetes", _route(), {"doc": "1"})

    assert hits == ["real-hit"]
    assert state["tier"] == "TIER1"
    assert calls == [("diabetes", 8, {"doc": "1"})]


def test_llm_citation_validation_does_not_depend_on_shared_system_hit_state():
    route = _route()
    compiled = {
        "claims": [EvidenceClaim("Diabetes is a metabolic disorder.", (2,), False, False, 0.95)],
        "compressed": "- Diabetes is a metabolic disorder. [S2]",
    }
    system = SimpleNamespace(_med_selected_hits=[])
    cascade = AnswerCascade(system)

    original_citation = AnswerCascade._citation_complete
    AnswerCascade._citation_complete = staticmethod(_citation_complete_without_shared_state(original_citation))
    try:
        original_generate = AnswerCascade.generate
        AnswerCascade.generate = _wrap_generate(original_generate)
        cascade._llm = lambda question, evidence, route: "Diabetes is a metabolic disorder. [S2]"
        answer, path, _ = cascade.generate("What is diabetes?", route, compiled)
    finally:
        AnswerCascade._citation_complete = original_citation
        AnswerCascade.generate = original_generate

    assert answer.endswith("[S2]")
    assert path == "PATH_C_CONSTRAINED_LLM"


def test_citation_guard_uses_current_generation_limit_not_shared_state():
    original = AnswerCascade._citation_complete
    guarded = _citation_complete_without_shared_state(original)
    import rag_project.runtime_functionality_deep_fix as fix

    previous = getattr(fix._TLS, "citation_limit", None)
    fix._TLS.citation_limit = 2
    try:
        assert guarded("A grounded medical statement. [S2]", 0) is True
    finally:
        if previous is None:
            del fix._TLS.citation_limit
        else:
            fix._TLS.citation_limit = previous


def test_followup_guard_does_not_misclassify_compound_question():
    def original(self, question, context, safety):
        return replace_route(_route(), is_follow_up=True)

    wrapped = _wrap_route(original)
    route = wrapped(SimpleNamespace(), "What causes cirrhosis and how is it managed?", "", SimpleNamespace())
    assert route.is_follow_up is False


def test_followup_guard_preserves_explicit_followup():
    def original(self, question, context, safety):
        return replace_route(_route(), is_follow_up=False)

    wrapped = _wrap_route(original)
    route = wrapped(SimpleNamespace(), "And what about treatment?", "previous context", SimpleNamespace())
    assert route.is_follow_up is True


def replace_route(route: RouteMetadata, **changes) -> RouteMetadata:
    from dataclasses import replace

    return replace(route, **changes)


def test_short_evidence_sentences_are_not_discarded():
    assert _functionality_sentences("DKA.\nNo.") == ["DKA.", "No."]


def test_equivalent_units_are_not_reported_as_contradiction():
    original_result = {
        "has_contradiction": True,
        "conflicts": [{"left": ["1 g"], "right": ["1000 mg"]}],
        "agreement": 0.65,
    }

    guarded = _filter_false_numeric_contradictions(lambda claims: original_result)
    result = guarded([])

    assert result["has_contradiction"] is False
    assert result["conflicts"] == []
    assert result["agreement"] == 1.0


def test_real_numeric_contradiction_is_preserved():
    original_result = {
        "has_contradiction": True,
        "conflicts": [{"left": ["500 mg"], "right": ["1000 mg"]}],
        "agreement": 0.65,
    }

    guarded = _filter_false_numeric_contradictions(lambda claims: original_result)
    result = guarded([])

    assert result["has_contradiction"] is True
    assert result["conflicts"]
    assert result["agreement"] == 0.65
