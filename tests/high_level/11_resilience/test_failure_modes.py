import pytest

@pytest.mark.high_level

def test_answer__returns_controlled_result_when_dependency_is_unavailable(clean_system, monkeypatch):
    retriever = getattr(clean_system, "retriever", None)
    if retriever is None or not hasattr(retriever, "retrieve"):
        pytest.skip("retriever fault injection is unavailable")
    original = retriever.retrieve
    monkeypatch.setattr(retriever, "retrieve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic retrieval outage")))
    result = clean_system.answer("What is diabetes mellitus?")
    assert str(result.get("status") or "").upper() in {"ANSWER_UNAVAILABLE", "NOT_SUPPORTED", "SUCCESS_WITH_WARNINGS", "SYSTEM_NOT_READY"}
    monkeypatch.setattr(retriever, "retrieve", original)

@pytest.mark.high_level

def test_runtime__declares_bounded_failure_controls(clean_system):
    assert int(clean_system.settings.ollama_failure_circuit_threshold) >= 1
    assert float(clean_system.settings.ollama_circuit_open_seconds) >= 1.0
