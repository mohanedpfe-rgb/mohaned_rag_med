from __future__ import annotations

import json

from rag_project.generation.llm_client import OllamaLLMClient


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        assert decode_unicode is True
        events = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
            {"done": True, "eval_count": 2, "total_duration": 123.0},
        ]
        return [json.dumps(event) for event in events]

    def close(self):
        return None


def test_generate_stream_yields_incremental_tokens(monkeypatch):
    client = OllamaLLMClient("http://127.0.0.1:11434", "test-model")

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("rag_project.generation.llm_client.requests.post", fake_post)

    result = list(client.generate_stream("question"))

    assert result == ["Hello", " world"]
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["json"]["stream"] is True
    assert client.last_metrics == {"eval_count": 2, "total_duration": 123.0}
