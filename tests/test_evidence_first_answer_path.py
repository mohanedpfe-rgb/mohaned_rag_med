from types import SimpleNamespace


def _hit(text, score=0.9, page=1, chunk="chunk-1"):
    from rag_project.app.rag_system import RetrievalHit
    return RetrievalHit(
        "doc-1", text,
        {"document_id": "doc-1", "chunk_id": chunk, "file_name": "medical.pdf", "page_numbers": [page]},
        score, score, score,
    )


def _system(hits, llm=None):
    from rag_project.intelligence.god_mode_100 import enhanced_god_answer
    system = SimpleNamespace()
    system.settings = SimpleNamespace(top_k=4, temperature=0.0, context_token_budget=3200)
    system.retriever = SimpleNamespace(retrieve=lambda *args, **kwargs: list(hits))
    system.citation_manager = SimpleNamespace(
        build=lambda selected: [{"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]}],
        validate=lambda citations, selected: citations,
    )
    system.llm = llm
    system.conversation_memory = SimpleNamespace(history=[], prompt_context=lambda: "", add=lambda *args: None)
    system.enhance = enhanced_god_answer
    return system


def test_evidence_first_returns_real_source_sentence_without_llm():
    hit = _hit("Insulin lowers blood glucose by promoting glucose uptake. [This sentence is source text.]" )
    system = _system([hit], llm=None)
    result = system.enhance(system, "What does insulin do?", None)
    assert result["status"] == "SUCCESS"
    assert "Insulin lowers blood glucose" in result["answer"]
    assert result["evidence_first"] is True
    assert result["grounding"]["supported_ratio"] == 1.0
    assert result["generation_path"] == "deterministic_extractive"


def test_evidence_first_can_use_local_llm_only_after_retrieval():
    class FakeLLM:
        def generate(self, **kwargs):
            return "Insulin lowers blood glucose. [S1]"

    hit = _hit("Insulin lowers blood glucose by promoting glucose uptake.")
    system = _system([hit], llm=FakeLLM())
    result = system.enhance(system, "What does insulin do?", None)
    assert result["status"] == "SUCCESS"
    assert result["grounding"]["allow"] is True
    assert result["generation_path"] == "local_llm_grounded"


def test_evidence_first_does_not_convert_enrichment_failure_into_no_answer(monkeypatch):
    hit = _hit("Hyperkalemia may result from impaired renal potassium excretion.")
    system = _system([hit], llm=None)
    monkeypatch.setattr("rag_project.intelligence.god_mode_100.build_evidence_hierarchy", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("diagnostic failure")))
    result = system.enhance(system, "What can cause hyperkalemia?", None)
    assert result["status"] == "SUCCESS"
    assert "Hyperkalemia" in result["answer"]
