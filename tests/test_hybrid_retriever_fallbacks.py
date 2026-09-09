from __future__ import annotations

from rag_project.retrieval.hybrid_retriever import HybridRetriever


class FakeEmbedding:
    def __init__(self, vector=None, fail=False):
        self.vector = vector or [0.1, 0.2]
        self.fail = fail

    def embed_query(self, query):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return self.vector


class FakeStore:
    def __init__(self, lexical=None, vector=None, lexical_error=False, vector_error=False):
        self.lexical = lexical or {"ids": [["lex-1"]], "documents": [["lexical evidence"]], "metadatas": [[{"document_id": "d1"}]], "distances": [[0.1]]}
        self.vector = vector or {"ids": [["vec-1"]], "documents": [["vector evidence"]], "metadatas": [[{"document_id": "d2"}]], "distances": [[0.1]]}
        self.lexical_error = lexical_error
        self.vector_error = vector_error

    def search_lexical(self, query, count, where):
        if self.lexical_error:
            raise RuntimeError("lexical unavailable")
        return self.lexical

    def search(self, embedding, count, where):
        if self.vector_error:
            raise RuntimeError("vector unavailable")
        return self.vector


def test_vector_mode_falls_back_to_lexical_when_embedding_fails():
    retriever = HybridRetriever(FakeStore(), FakeEmbedding(fail=True), lexical_mode="vector")
    hits = retriever.retrieve("question", top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "d1"


def test_lexical_mode_falls_back_to_vector_when_lexical_fails():
    retriever = HybridRetriever(FakeStore(lexical_error=True), FakeEmbedding(), lexical_mode="lexical")
    hits = retriever.retrieve("question", top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "d2"


def test_hybrid_survives_single_branch_failure():
    retriever = HybridRetriever(FakeStore(vector_error=True), FakeEmbedding(), lexical_mode="hybrid")
    hits = retriever.retrieve("question", top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "d1"


def test_malformed_backend_lengths_are_ignored_safely():
    store = FakeStore(
        lexical={"ids": [["lex-1", "lex-2"]], "documents": [["only one"]], "metadatas": [[]], "distances": [[]]},
        vector={"ids": [["vec-1"]], "documents": [["vector"]], "metadatas": [[]], "distances": [[]]},
    )
    retriever = HybridRetriever(store, FakeEmbedding(), lexical_mode="hybrid")
    hits = retriever.retrieve("question", top_k=5)
    assert [hit.text for hit in hits]


def test_invalid_top_k_and_empty_query_are_safe():
    retriever = HybridRetriever(FakeStore(), FakeEmbedding())
    assert retriever.retrieve("", top_k=5) == []
    try:
        retriever.retrieve("question", top_k="bad")
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("invalid top_k must raise ValueError")
