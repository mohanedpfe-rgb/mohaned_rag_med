from types import SimpleNamespace


def _settings():
    return SimpleNamespace(
        top_k=4,
        max_query_variants=3,
        retrieval_candidate_multiplier=2,
    )


def test_safe_hits_treats_none_retriever_result_as_empty():
    from rag_project.intelligence.god_mode import _safe_hits
    from rag_project.intelligence.query_intelligence import plan_query

    class Retriever:
        def retrieve(self, *args, **kwargs):
            return None

    system = SimpleNamespace(
        settings=_settings(),
        retriever=Retriever(),
        reranker=SimpleNamespace(rerank=lambda *args, **kwargs: None),
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    result = _safe_hits(system, plan_query("What are the main findings?"), None)
    assert result == []


def test_safe_hits_treats_none_reranker_result_as_failed_reranking():
    from rag_project.app.rag_system import RetrievalHit
    from rag_project.intelligence.god_mode import _safe_hits
    from rag_project.intelligence.query_intelligence import plan_query

    hit = RetrievalHit(
        "doc-1",
        "Diabetes is chronic.",
        {"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]},
        0.9,
        0.9,
        0.9,
    )

    class Retriever:
        def retrieve(self, *args, **kwargs):
            return [hit]

    system = SimpleNamespace(
        settings=_settings(),
        retriever=Retriever(),
        reranker=SimpleNamespace(rerank=lambda *args, **kwargs: None),
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    result = _safe_hits(system, plan_query("What is diabetes?"), None)
    assert result
    assert result[0].text == "Diabetes is chronic."


def test_none_items_inside_retriever_result_are_ignored():
    from rag_project.app.rag_system import RetrievalHit
    from rag_project.intelligence.god_mode import _safe_hits
    from rag_project.intelligence.query_intelligence import plan_query

    hit = RetrievalHit(
        "doc-1",
        "Diabetes is chronic.",
        {"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]},
        0.9,
        0.9,
        0.9,
    )

    class Retriever:
        def retrieve(self, *args, **kwargs):
            return [None, hit]

    system = SimpleNamespace(
        settings=_settings(),
        retriever=Retriever(),
        reranker=SimpleNamespace(rerank=lambda *args, **kwargs: None),
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    result = _safe_hits(system, plan_query("What is diabetes?"), None)
    assert len(result) == 1
    assert result[0].text == "Diabetes is chronic."
