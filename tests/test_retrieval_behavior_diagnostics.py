from __future__ import annotations

import math

import pytest

from rag_project.retrieval.hybrid_retriever import HybridRetriever


class FakeEmbedding:
    def __init__(self, value=None):
        self.value = value if value is not None else [1.0, 0.0, 0.0]
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return self.value


class FakeStore:
    def __init__(self, vector=None, lexical=None, vector_error=None, lexical_error=None):
        self.vector = vector or {"ids": [], "documents": [], "metadatas": [], "distances": []}
        self.lexical = lexical or {"ids": [], "documents": [], "metadatas": [], "distances": []}
        self.vector_error = vector_error
        self.lexical_error = lexical_error
        self.calls = []

    def search(self, embedding, n_results, where=None):
        self.calls.append(("vector", embedding, n_results, where))
        if self.vector_error:
            raise self.vector_error
        return self.vector

    def search_lexical(self, query, n_results, where=None):
        self.calls.append(("lexical", query, n_results, where))
        if self.lexical_error:
            raise self.lexical_error
        return self.lexical


def results(*rows, nested=False):
    ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    metas = [r[2] for r in rows]
    distances = [r[3] for r in rows]
    if nested:
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [distances]}
    return {"ids": ids, "documents": docs, "metadatas": metas, "distances": distances}


def test_empty_query_returns_empty_without_backend_calls():
    embedding = FakeEmbedding()
    store = FakeStore()
    retriever = HybridRetriever(store, embedding)
    assert retriever.retrieve("   ") == []
    assert embedding.calls == []
    assert store.calls == []


def test_top_k_is_clamped_to_safe_range():
    row = ("a", "text", {"document_id": "d"}, 0.1)
    store = FakeStore(vector=results(row), lexical=results(row))
    retriever = HybridRetriever(store, FakeEmbedding())
    assert len(retriever.retrieve("query", top_k=0)) <= 1
    assert len(retriever.retrieve("query", top_k=999)) <= 1


def test_top_k_invalid_type_raises_clear_error():
    retriever = HybridRetriever(FakeStore(), FakeEmbedding())
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k="not-an-int")


@pytest.mark.parametrize("mode", ["hybrid", "lexical", "vector"])
def test_valid_retrieval_modes_construct(mode):
    retriever = HybridRetriever(FakeStore(), FakeEmbedding(), lexical_mode=mode, vector_weight=0.7)
    assert retriever.lexical_mode == mode


def test_invalid_retrieval_mode_rejected():
    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        HybridRetriever(FakeStore(), FakeEmbedding(), lexical_mode="bm25", vector_weight=0.7)


@pytest.mark.parametrize("weight, expected", [(-10.0, 0.0), (0.5, 0.5), (10.0, 1.0)])
def test_vector_weight_is_clamped(weight, expected):
    retriever = HybridRetriever(FakeStore(), FakeEmbedding(), lexical_mode="hybrid", vector_weight=weight)
    assert retriever.vector_weight == expected


@pytest.mark.parametrize("weight", [math.inf, -math.inf, math.nan])
def test_non_finite_vector_weight_rejected(weight):
    with pytest.raises(ValueError, match="finite"):
        HybridRetriever(FakeStore(), FakeEmbedding(), vector_weight=weight)


def test_vector_query_calls_embedding_exactly_once():
    row = ("a", "insulin", {"document_id": "d"}, 0.1)
    embedding = FakeEmbedding()
    store = FakeStore(vector=results(row))
    retriever = HybridRetriever(store, embedding, lexical_mode="vector")
    hits = retriever.retrieve("  insulin  ", top_k=1)
    assert hits
    assert embedding.calls == ["insulin"]
    assert [c[0] for c in store.calls] == ["vector"]


def test_hybrid_query_runs_vector_and_lexical_branches():
    row = ("a", "insulin", {"document_id": "d"}, 0.1)
    store = FakeStore(vector=results(row), lexical=results(row))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="hybrid")
    hits = retriever.retrieve("insulin")
    assert hits
    assert {c[0] for c in store.calls} == {"vector", "lexical"}


def test_metadata_filter_reaches_both_branches():
    row = ("a", "insulin", {"document_id": "d"}, 0.1)
    store = FakeStore(vector=results(row), lexical=results(row))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="hybrid")
    where = {"document_id": "d"}
    retriever.retrieve("insulin", where=where)
    assert all(call[3] == where for call in store.calls)


def test_nested_chroma_style_results_are_unpacked():
    row = ("a", "insulin", {"document_id": "d"}, 0.2)
    store = FakeStore(vector=results(row, nested=True))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="vector")
    hits = retriever.retrieve("insulin", top_k=1)
    assert len(hits) == 1
    assert hits[0].text == "insulin"


def test_missing_metadata_and_distances_do_not_crash():
    store = FakeStore(vector={"ids": ["a", "b"], "documents": ["a", "b"], "metadatas": [{}], "distances": []})
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="vector")
    hits = retriever.retrieve("a")
    assert len(hits) == 2
    assert all(math.isfinite(h.score) for h in hits)


def test_nonfinite_distances_are_safely_demoted():
    row_good = ("a", "good", {"document_id": "d"}, 0.1)
    row_bad = ("b", "bad", {"document_id": "d"}, float("nan"))
    store = FakeStore(vector=results(row_good, row_bad))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="vector")
    hits = retriever.retrieve("good", top_k=2)
    assert hits[0].metadata.get("document_id") == "d"
    assert all(math.isfinite(h.vector_score) for h in hits)


def test_negative_distance_is_clamped_before_exponential_transform():
    row = ("a", "good", {"document_id": "d"}, -3.0)
    store = FakeStore(vector=results(row))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="vector")
    hit = retriever.retrieve("good")[0]
    assert hit.vector_score == pytest.approx(1.0)


def test_vector_branch_failure_falls_back_to_lexical_in_vector_mode():
    vector_row = ("v", "vector", {"document_id": "v"}, 0.1)
    lexical_row = ("l", "insulin resistance", {"document_id": "l"}, 0.1)
    store = FakeStore(vector_error=RuntimeError("vector unavailable"), lexical=results(lexical_row))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="vector")
    hits = retriever.retrieve("insulin resistance", top_k=1)
    assert hits and hits[0].metadata["document_id"] == "l"
    assert any(c[0] == "lexical" for c in store.calls)


def test_lexical_branch_failure_falls_back_to_vector_in_lexical_mode():
    vector_row = ("v", "insulin resistance", {"document_id": "v"}, 0.1)
    store = FakeStore(vector=results(vector_row), lexical_error=RuntimeError("fts unavailable"))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="lexical")
    hits = retriever.retrieve("insulin resistance", top_k=1)
    assert hits and hits[0].metadata["document_id"] == "v"


def test_both_backend_failures_fail_closed_to_empty():
    store = FakeStore(vector_error=RuntimeError("vector"), lexical_error=RuntimeError("lexical"))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="hybrid")
    assert retriever.retrieve("query") == []


def test_duplicate_id_from_both_sources_merges_scores_into_one_hit():
    row_v = ("same", "vector text", {"document_id": "d"}, 0.1)
    row_l = ("same", "lexical text", {"document_id": "d"}, 0.2)
    store = FakeStore(vector=results(row_v), lexical=results(row_l))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="hybrid")
    hits = retriever.retrieve("vector text")
    assert len(hits) == 1
    assert hits[0].vector_score > 0
    assert hits[0].lexical_score > 0


def test_lexical_mode_still_populates_lexical_score():
    row = ("a", "hypokalemia potassium", {"document_id": "d"}, 0.1)
    store = FakeStore(lexical=results(row))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="lexical")
    hit = retriever.retrieve("hypokalemia potassium")[0]
    assert hit.lexical_score > 0
    assert hit.score == hit.lexical_score


def test_vector_mode_populates_vector_score():
    row = ("a", "hypokalemia", {"document_id": "d"}, 0.3)
    store = FakeStore(vector=results(row))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="vector")
    hit = retriever.retrieve("hypokalemia")[0]
    assert hit.vector_score > 0


def test_hybrid_scores_are_finite_and_nonnegative():
    rows = [
        ("a", "insulin", {"document_id": "a"}, 0.1),
        ("b", "resistance", {"document_id": "b"}, 0.8),
    ]
    store = FakeStore(vector=results(*rows), lexical=results(*rows))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="hybrid")
    hits = retriever.retrieve("insulin resistance", top_k=2)
    assert all(math.isfinite(h.score) and h.score >= 0 for h in hits)


def test_query_tokenization_does_not_turn_whitespace_into_backend_query():
    row = ("a", "diabetes insulin", {"document_id": "d"}, 0.1)
    store = FakeStore(lexical=results(row))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="lexical")
    retriever.retrieve(" diabetes   insulin ")
    lexical_queries = [c[1] for c in store.calls if c[0] == "lexical"]
    assert lexical_queries == ["diabetes insulin"]


def test_results_are_sorted_descending_by_score():
    rows = [
        ("a", "weak", {"document_id": "a"}, 0.9),
        ("b", "strong strong", {"document_id": "b"}, 0.1),
    ]
    store = FakeStore(lexical=results(*rows))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="lexical")
    hits = retriever.retrieve("strong strong", top_k=2)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_retrieve_returns_at_most_top_k_results():
    rows = [(str(i), f"text {i}", {"document_id": str(i)}, 0.1 + i / 100) for i in range(20)]
    store = FakeStore(vector=results(*rows))
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="vector")
    assert len(retriever.retrieve("text", top_k=7)) == 7
