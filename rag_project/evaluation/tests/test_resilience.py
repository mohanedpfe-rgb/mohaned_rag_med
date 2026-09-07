import time
import pytest
import requests
from unittest.mock import patch, MagicMock

from rag_project.generation.llm_client import OllamaLLMClient


@patch("time.sleep", return_value=None)
def test_retry_success_after_two_transient_failures(mock_sleep):
    """Monkeypatch requests.post to raise ConnectionError twice then return valid response.
    Call OllamaLLMClient.generate, assert result is the good answer and client.retries_encountered >= 2."""
    client = OllamaLLMClient(base_url="http://fake", model="fake", timeout_seconds=5)

    good_response = MagicMock(status_code=200)
    good_response.json = lambda: {"message": {"content": "ok answer"}}

    side_effects = [
        requests.ConnectionError("boom1"),
        requests.ConnectionError("boom2"),
        good_response,
    ]

    with patch.object(requests, "post", side_effect=side_effects) as mock_post:
        result = client.generate("hello prompt")

    assert result.strip() == "ok answer"
    assert client.retries_encountered >= 2
    assert mock_post.call_count == 3


@patch("time.sleep", return_value=None)
def test_retry_exhausted_propagates_runtime_error(mock_sleep):
    """All 5 attempts fail with ConnectionError. Assert RuntimeError is raised, not silently swallowed."""
    client = OllamaLLMClient(base_url="http://fake", model="fake", timeout_seconds=5)

    side_effects = [requests.ConnectionError(f"boom{i}") for i in range(10)]

    with patch.object(requests, "post", side_effect=side_effects):
        with pytest.raises(RuntimeError) as exc_info:
            client.generate("hello prompt")

    assert "Generation service failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, requests.ConnectionError)
    assert client.retries_encountered >= 4


def test_generation_payload_has_output_budget():
    client = OllamaLLMClient(
        base_url="http://fake",
        model="fake",
        timeout_seconds=5,
        max_output_tokens=123,
    )
    response = MagicMock(status_code=200)
    response.json = lambda: {"message": {"content": "ok"}}
    with patch.object(requests, "post", return_value=response) as mock_post:
        assert client.generate("hello") == "ok"
    assert mock_post.call_args.kwargs["json"]["options"]["num_predict"] == 123
