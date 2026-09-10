from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag_project.app.production_rag import ProductionRAGSystem
from rag_project.app.rag_system import EvidenceAlignment, QueryQualityClassifier
from rag_project.intelligence.god_mode import _safe_hits
from rag_project.retrieval.hybrid_retriever import RetrievalHit


@dataclass
class _FakePlan:
    normalized: str = "what is diabetic nephropathy"
    variants: tuple[str, ...] = ("diabetic nephropathy", "nephropathy in diabetes")
    subqueries: tuple[str, ...] = ()
    entities: tuple[str, ...] = ("diabetic", "nephropathy")
    needs_table: bool = False
    needs_figure: bool = False
    needs_multi_hop: bool = False
    score_boosts: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.score_boosts is None:
            self.score_boosts = {"table": 1.2, "figure": 1.2, "parent": 1.1, "entity": 1.05}


def _hit(i: int, text: str | None = None, score: float = 0.8) -> RetrievalHit:
    return RetrievalHit(
        doc_id=f"doc-{i % 3}",
        text=text if text is not None else f"Evidence about diabetic nephropathy and albuminuria {i}",
        metadata={"document_id": f"doc-{i % 3}", "chunk_id": f"chunk-{i}", "page_numbers": [i + 1], "language": "en"},
        score=score,
        vector_score=0.65,
        lexical_score=0.2,
    )


def _fake_system(retrieved_hits: list[RetrievalHit], *, max_query_variants: int = 8):
    calls: list[tuple[str, int, object]] = []

    class Retriever:
        def retrieve(self, query, top_k=6, where=None):
            calls.append((query, top_k, where))
            return list(retrieved_hits[:top_k])

    class Reranker:
        def rerank(self, query, hits):
            return list(hits)

    logger = SimpleNamespace(warning=lambda *a, **k: None)
    settings = SimpleNamespace(top_k=6, retrieval_candidate_multiplier=5, max_query_variants=max_query_variants)
    system = SimpleNamespace(retriever=Retriever(), reranker=Reranker(), settings=settings, logger=logger)
    return system, calls


def test_production_certifier_has_bound_method_signature_not_extra_self():
    import inspect

    signature = inspect.signature(ProductionRAGSystem._certified_god_answer)
    assert list(signature.parameters) == ["self", "question", "metadata_filter"]


def test_production_answer_passes_self_only_once(monkeypatch):
    system = ProductionRAGSystem.__new__(ProductionRAGSystem)
    system._production_feature_contract = {"all_resolved": True}
    system.settings = SimpleNamespace()
    seen = {}

    def fake_certifier(self, question, metadata_filter=None):
        seen["question"] = question
        seen["filter"] = metadata_filter
        return {"status": "OK", "answer": "supported", "hits": [], "confidence": {}}

    monkeypatch.setattr(ProductionRAGSystem, "_certified_god_answer", fake_certifier)
    monkeypatch.setattr("rag_project.app.production_rag.apply_medical_safety_policy", lambda q, r, s: r)
    monkeypatch.setattr("rag_project.app.production_rag.sanitize_trace", lambda x: x)

    result = system.answer("What is supported?", {"document_id": "doc-1"})
    assert result["answer"] == "supported"
    assert seen == {"question": "What is supported?", "filter": {"document_id": "doc-1"}}


def test_query_quality_rejects_empty_whitespace():
    result = QueryQualityClassifier.assess("   \n\t  ")
    assert result["should_abstain"] is True
    assert result["query_quality"] == "LOW_QUALITY_QUERY"


@pytest.mark.parametrize("query", ["related", "relation", "and or", "plus", "sont"])
def test_query_quality_rejects_low_signal_fragments(query):
    assert QueryQualityClassifier.assess(query)["should_abstain"] is True


def test_query_quality_accepts_specific_medical_question():
    result = QueryQualityClassifier.assess("What are the diagnostic criteria of diabetic ketoacidosis?")
    assert result["should_abstain"] is False
    assert result["query_intent"] == "DIRECT_QUESTION"


def test_evidence_alignment_rejects_topic_only_match():
    hits = [_hit(1, "Thyroid cancer can be papillary and associated with RET mutations.")]
    result = EvidenceAlignment.evaluate("What are the causes of diabetic nephropathy?", hits)
    assert result["decision"] in {"RELATED_BUT_NOT_ANSWERING", "NOT_SUPPORTED"}
    assert result["local_context_strength"] < 0.2 or result["answerability"] < 0.35


def test_evidence_alignment_accepts_direct_french_support():
    hits = [_hit(1, "La néphropathie diabétique est une complication microvasculaire chronique du diabète.")]
    result = EvidenceAlignment.evaluate("Qu'est-ce que la néphropathie diabétique ?", hits)
    assert result["decision"] in {"DIRECTLY_SUPPORTED", "PARTIALLY_SUPPORTED"}
    assert result["answerability"] > 0.0


def test_evidence_alignment_handles_empty_hit_text():
    hits = [SimpleNamespace(text="")]
    result = EvidenceAlignment.evaluate("What is diabetic nephropathy?", hits)
    assert result["answerability"] == 0.0
    assert result["decision"] in {"RELATED_BUT_NOT_ANSWERING", "NOT_SUPPORTED"}


def test_evidence_alignment_does_not_depend_only_on_number_of_hits():
    one = EvidenceAlignment.evaluate("What is insulin resistance?", [_hit(1, "Insulin resistance is reduced biological response to insulin.")])
    many = EvidenceAlignment.evaluate("What is insulin resistance?", [_hit(i, "Unrelated thyroid cancer facts.") for i in range(1, 9)])
    assert one["decision"] != "RELATED_BUT_NOT_ANSWERING"
    assert many["decision"] in {"RELATED_BUT_NOT_ANSWERING", "NOT_SUPPORTED"}


def test_safe_hits_deduplicates_same_chunk_from_multiple_variants():
    duplicate = _hit(1)
    system, calls = _fake_system([duplicate, duplicate])
    plan = _FakePlan()
    hits = _safe_hits(system, plan, None)
    assert len({h.metadata["chunk_id"] for h in hits}) == len(hits)
    assert len(calls) == 3  # normalized query + two distinct variants


def test_safe_hits_respects_query_variant_limit():
    plan = _FakePlan(variants=tuple(f"variant {i}" for i in range(20)))
    system, calls = _fake_system([_hit(i) for i in range(40)], max_query_variants=3)
    _safe_hits(system, plan, None)
    assert len(calls) == 3


def test_safe_hits_propagates_metadata_filter_to_every_variant():
    plan = _FakePlan(variants=("a", "b"))
    system, calls = _fake_system([_hit(1)], max_query_variants=8)
    where = {"document_id": "doc-1"}
    _safe_hits(system, plan, where)
    assert calls
    assert all(call[2] == where for call in calls)


def test_safe_hits_candidate_budget_tripwire():
    plan = _FakePlan(variants=tuple(f"variant {i}" for i in range(7)))
    many = [_hit(i) for i in range(60)]
    seen = []

    class Retriever:
        def retrieve(self, query, top_k=6, where=None):
            seen.append(top_k)
            return list(many[:top_k])

    class Reranker:
        def rerank(self, query, hits):
            return list(hits)

    system = SimpleNamespace(
        retriever=Retriever(),
        reranker=Reranker(),
        settings=SimpleNamespace(top_k=6, retrieval_candidate_multiplier=5, max_query_variants=8),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
    )
    _safe_hits(system, plan, None)
    assert sum(seen) <= 48


def test_safe_hits_should_preserve_a_strong_retrieval_signal_after_reranking():
    hit = _hit(1, score=0.9)
    hit.vector_score = 0.9

    class Retriever:
        def retrieve(self, query, top_k=6, where=None):
            return [hit]

    class Reranker:
        def rerank(self, query, hits):
            hits[0].score = 0.0001
            return hits

    system = SimpleNamespace(
        retriever=Retriever(),
        reranker=Reranker(),
        settings=SimpleNamespace(top_k=1, retrieval_candidate_multiplier=5, max_query_variants=1),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
    )
    result = _safe_hits(system, _FakePlan(), None)
    assert result[0].score >= 0.45


def test_safe_hits_survives_reranker_exception():
    hit = _hit(1)

    class Retriever:
        def retrieve(self, query, top_k=6, where=None):
            return [hit]

    class Reranker:
        def rerank(self, query, hits):
            raise RuntimeError("model failure")

    system = SimpleNamespace(
        retriever=Retriever(), reranker=Reranker(),
        settings=SimpleNamespace(top_k=1, retrieval_candidate_multiplier=5, max_query_variants=1),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
    )
    result = _safe_hits(system, _FakePlan(), None)
    assert result and result[0].metadata["chunk_id"] == "chunk-1"


def test_safe_hits_survives_one_failed_query_variant():
    calls = []

    class Retriever:
        def retrieve(self, query, top_k=6, where=None):
            calls.append(query)
            if query == "bad":
                raise RuntimeError("branch failed")
            return [_hit(1)]

    class Reranker:
        def rerank(self, query, hits):
            return hits

    system = SimpleNamespace(
        retriever=Retriever(), reranker=Reranker(),
        settings=SimpleNamespace(top_k=1, retrieval_candidate_multiplier=5, max_query_variants=2),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
    )
    plan = _FakePlan(variants=("bad", "good"))
    result = _safe_hits(system, plan, None)
    assert calls == [plan.normalized, "bad"]
    assert result


def test_safe_hits_limits_document_concentration():
    many = [_hit(i, score=1.0 - i / 1000.0) for i in range(20)]
    for hit in many:
        hit.doc_id = "same-doc"
        hit.metadata["document_id"] = "same-doc"

    class Retriever:
        def retrieve(self, query, top_k=6, where=None):
            return many[:top_k]

    class Reranker:
        def rerank(self, query, hits):
            return hits

    system = SimpleNamespace(
        retriever=Retriever(), reranker=Reranker(),
        settings=SimpleNamespace(top_k=6, retrieval_candidate_multiplier=5, max_query_variants=1),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
    )
    result = _safe_hits(system, _FakePlan(), None)
    assert len(result) <= 12
    assert sum(1 for h in result if h.metadata["document_id"] == "same-doc") <= 12
