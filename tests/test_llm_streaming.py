from __future__ import annotations

import json

import pytest

from rag_project.generation.llm_client import OllamaLLMClient


class FakeResponse:
    status_code = 200

    def __init__(self, events):
        self.events = events

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        assert decode_unicode is True
        return [json.dumps(event) for event in self.events]

    def close(self):
        return None


def test_generate_stream_yields_incremental_tokens(monkeypatch):
    client = OllamaLLMClient("http://127.0.0.1:11434", "test-model")

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse([
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
            {"done": True, "eval_count": 2, "total_duration": 123.0},
        ])

    monkeypatch.setattr("rag_project.generation.llm_client.requests.post", fake_post)

    result = list(client.generate_stream("question"))

    assert result == ["Hello", " world"]
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["json"]["stream"] is True
    assert client.last_metrics == {"eval_count": 2, "total_duration": 123.0}


def test_generate_stream_rejects_truncated_connection(monkeypatch):
    client = OllamaLLMClient("http://127.0.0.1:11434", "test-model")

    monkeypatch.setattr(
        "rag_project.generation.llm_client.requests.post",
        lambda *args, **kwargs: FakeResponse([{"message": {"content": "partial"}}]),
    )

    with pytest.raises(RuntimeError, match="streaming generation failed"):
        list(client.generate_stream("question"))
