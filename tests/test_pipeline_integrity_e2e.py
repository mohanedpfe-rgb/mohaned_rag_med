from types import SimpleNamespace

from rag_project.intelligence.pipeline_integrity import install
from rag_project.intelligence.top_level_pipeline import complete_phases


class _Memory:
    history = []

    @staticmethod
    def prompt_context():
        return ""


class _Settings:
    top_k = 6
    context_token_budget = 3200
    temperature = 0.2


class _System:
    settings = _Settings()
    llm = None
    conversation_memory = _Memory()



def _hit(text: str, score: float = 0.82):
    return SimpleNamespace(
        text=text,
        score=score,
        doc_id="doc-1",
        metadata={"document_id": "doc-1", "chunk_id": "chunk-1", "page_numbers": [1]},
    )


def test_canonical_simple_query_reaches_direct_summary_without_fake_hardening():
    install()
    system = _System()
    result = complete_phases(
        system,
        "What are the main findings?",
        {"hits": [_hit("Les principales conséquences biologiques sont l'hyperglycémie, la cétose et l'acidose métabolique.")]},
    )

    assert result["rewritten_question"] == "What are the main findings?"
    assert result["phase_plan"]["intent"] == "factual"
    assert result["phase_plan"]["entities"] == ()
    assert result["phase_plan"]["needs_multi_hop"] is False
    assert result["two_stage_policy"]["required"] is False
    assert result["extractive_stage"]["supported"] is True
    assert result["answer"]
    assert result["status"] != "GENERATION_ABSTAIN"


def test_canonical_simple_query_does_not_contain_internal_protocol_text():
    install()
    system = _System()
    result = complete_phases(
        system,
        "What are the main findings?",
        {"hits": [_hit("The main findings are clearly enumerated in the document.")]},
    )
    query = result["rewritten_question"].casefold()
    assert "relevant entities:" not in query
    assert "follow-up:" not in query
    assert "planner:" not in query
    assert "intent:" not in query
