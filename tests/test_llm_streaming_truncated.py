from __future__ import annotations

import json

import pytest

from rag_project.generation.llm_client import OllamaLLMClient


class TruncatedResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        assert decode_unicode is True
        return [json.dumps({"message": {"content": "partial"}})]

    def close(self):
        return None


def test_generate_stream_rejects_truncated_response(monkeypatch):
    client = OllamaLLMClient("http://127.0.0.1:11434", "test-model")
    monkeypatch.setattr(
        "rag_project.generation.llm_client.requests.post",
        lambda *_args, **_kwargs: TruncatedResponse(),
    )
    with pytest.raises(RuntimeError, match="streaming generation failed"):
        list(client.generate_stream("question"))
    assert client.last_error == "RuntimeError"
