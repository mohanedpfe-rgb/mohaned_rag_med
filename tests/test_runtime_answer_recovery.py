from types import SimpleNamespace


def _hit():
    from rag_project.app.rag_system import RetrievalHit

    return RetrievalHit(
        "doc-1",
        "Diabetes mellitus is a chronic disease characterized by hyperglycemia.",
        {"document_id": "doc-1", "chunk_id": "chunk-1", "file_name": "medical.pdf", "page_numbers": [1]},
        0.9,
        0.9,
        0.9,
    )


def _bare_system(retrieve_result):
    from rag_project.app.production_rag import ProductionRAGSystem

    system = ProductionRAGSystem.__new__(ProductionRAGSystem)
    system.settings = SimpleNamespace(top_k=4)
    system.retriever = SimpleNamespace(retrieve=lambda *args, **kwargs: retrieve_result)
    system.citation_manager = SimpleNamespace(
        build=lambda hits: [{"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]}],
        validate=lambda citations, hits: citations,
    )
    system.logger = SimpleNamespace(exception=lambda *args, **kwargs: None)
    return system


def test_context_builder_accepts_none_hits():
    from rag_project.retrieval.context_builder import ContextBuilder

    context, selected = ContextBuilder().build(None)
    assert context == ""
    assert selected == []


def test_context_builder_ignores_none_neighbor_result():
    from rag_project.retrieval.context_builder import ContextBuilder

    hit = _hit()
    builder = ContextBuilder(neighbor_expansion=True, neighbor_resolver=lambda *args: None)
    context, selected = builder.build([hit])
    assert selected == [hit]
    assert "Diabetes mellitus" in context


def test_runtime_recovery_handles_none_retrieval_and_does_not_raise():
    system = _bare_system(None)
    result = system._recovery_answer("What is diabetes?", None, TypeError("NoneType is not iterable"))
    assert result["status"] == "NOT_SUPPORTED"
    assert result["hits"] == []
    assert result["recovery"]["attempted"] is True


def test_runtime_recovery_returns_grounded_extractive_answer():
    hit = _hit()
    system = _bare_system([hit])
    result = system._recovery_answer("What is diabetes?", None, RuntimeError("downstream failure"))
    assert result["status"] == "SUCCESS_WITH_WARNINGS"
    assert result["answer"]
    assert "[S1]" in result["answer"]
    assert result["recovery"]["grounded_extractive_fallback"] is True


def test_runtime_answer_catches_certified_pipeline_failure_and_uses_recovery(monkeypatch):
    from rag_project.app import production_rag

    system = _bare_system([_hit()])
    system._production_feature_contract = {"all_resolved": True}
    system.conversation_memory = SimpleNamespace(history=[], add=lambda *args: None)
    system._certified_god_answer = lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("NoneType object is not iterable"))
    monkeypatch.setattr(production_rag, "apply_medical_safety_policy", lambda q, result, settings: result)
    monkeypatch.setattr(production_rag, "request_budget", lambda settings: __import__("contextlib").nullcontext(45.0))
    monkeypatch.setattr(production_rag, "elapsed", lambda: 0.01)
    monkeypatch.setattr(production_rag, "exhausted", lambda: False)

    result = system.answer("What is diabetes?")
    assert result["status"] == "SUCCESS_WITH_WARNINGS"
    assert result["answer"]
    assert result["recovery"]["grounded_extractive_fallback"] is True
