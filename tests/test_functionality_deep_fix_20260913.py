from __future__ import annotations

from types import SimpleNamespace

from rag_project.intelligence.evidence_guard import verify_claims
from rag_project.intelligence.med_evidence_pro import RetrievalHit
from rag_project.runtime_deep_contract_fix import _cache_observability_wrapper, _unit_aware_numeric_verifier


class _FakeCache:
    def __init__(self):
        self.deleted: list[str] = []

    def get(self, _query):
        return None

    def restore(self, payload):
        return payload

    def delete(self, query):
        self.deleted.append(query)


class _FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, question, top_k, where):
        self.calls.append((question, top_k, where))
        return []


def test_filtered_retrieval_never_leaves_filtered_hits_in_global_cache():
    cache = _FakeCache()
    retriever = _FakeRetriever()

    class _Self:
        def __init__(self):
            self.cache = cache
            self.system = SimpleNamespace(retriever=retriever)

        @staticmethod
        def _confidence(_hits, _entities):
            return 0.0

    def original(self, question, route, where):
        assert where == {"document_id": "target"}
        return [RetrievalHit("target", "target evidence", {"document_id": "target"}, 1.0)], {"tier": "TIER1"}

    wrapped = _cache_observability_wrapper(original)
    self = _Self()
    result, _meta = wrapped(self, "same question", SimpleNamespace(entities=()), {"document_id": "target"})

    assert result[0].doc_id == "target"
    assert cache.deleted == ["same question"]
    assert retriever.calls == [("same question", 1, {"document_id": "target"})]


def test_unit_equivalent_numeric_values_are_not_reported_as_raw_string_mismatch():
    answer = "The dose is 1 g [S1]."
    hits = [RetrievalHit("doc", "The dose is 1000 mg.", {"document_id": "doc"}, 1.0)]
    semantic_checks = list(verify_claims(answer, [hits[0].text], ["S1"]))
    assert not any(
        getattr(check, "numeric_mismatch", False) or getattr(check, "status", "") == "NUMERIC_MISMATCH"
        for check in semantic_checks
    )

    class _Verifier:
        pass

    def original(self, answer, hits, route, compiled):
        return {
            "allow": False,
            "checked": True,
            "grounding": {"allow": True},
            "final_answer": {"allow": True},
            "numeric_mismatch": True,
        }

    wrapped = _unit_aware_numeric_verifier(original)
    result = wrapped(
        _Verifier(),
        answer,
        hits,
        SimpleNamespace(numeric_sensitivity=True),
        {"numeric_values": ["1000 mg"]},
    )

    assert result["numeric_mismatch"] is False
    assert result["allow"] is True
