from __future__ import annotations

import importlib

import numpy as np


def test_medical_output_handles_numpy_citations_without_truthiness_error():
    security = importlib.import_module("rag_project.security")
    result = security.postprocess_medical_output(
        {
            "answer": "The dose is 10 mg twice daily.",
            "citations": np.array([], dtype=object),
        }
    )
    assert result["safety_backstop"] == "medical_action_without_citation"


def test_medical_output_preserves_cited_high_risk_answer_with_numpy_citations():
    security = importlib.import_module("rag_project.security")
    result = security.postprocess_medical_output(
        {
            "answer": "The dose is 10 mg twice daily.",
            "citations": np.array([{"page": 10}], dtype=object),
        }
    )
    assert "safety_backstop" not in result


def test_runtime_v8_evidence_normalization_rejects_array_truthiness():
    from rag_project.runtime_stability_v8 import _safe_evidence

    result = {"evidence": np.array([{"text": "A"}, {"text": "B"}], dtype=object)}
    evidence = _safe_evidence(result)
    assert type(evidence) is list
    assert len(evidence) == 2


def test_runtime_v8_evidence_uses_fallback_without_array_truthiness():
    from rag_project.runtime_stability_v8 import _safe_evidence

    result = {"evidence": None, "hits": np.array([1, 2, 3])}
    assert _safe_evidence(result) == [1, 2, 3]


def test_runtime_v8_ollama_health_rejects_unsafe_urls_before_network(monkeypatch):
    from rag_project.runtime_stability_v8 import _safe_ollama_health

    called = False

    class FakeRequests:
        def get(self, *args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("network should not be attempted for an unsafe URL")

    import requests
    monkeypatch.setattr(requests, "get", FakeRequests().get)
    ok, message, models = _safe_ollama_health(lambda *_args, **_kwargs: None, "http://169.254.169.254")
    assert not ok
    assert message == "ValueError"
    assert models == []
    assert not called
