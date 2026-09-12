from __future__ import annotations

import pytest

from rag_project.security import MAX_QUERY_CHARS, validate_ollama_url, validate_query


@pytest.mark.high_level
def test_security__ollama_url_rejects_embedded_credentials():
    with pytest.raises(ValueError, match="without embedded credentials"):
        validate_ollama_url("http://user:pass@localhost:11434")


@pytest.mark.high_level
def test_security__remote_ollama_hostname_requires_explicit_allowlist(monkeypatch):
    monkeypatch.delenv("BOOKRAG_OLLAMA_ALLOWLIST", raising=False)
    with pytest.raises(ValueError, match="allowlist"):
        validate_ollama_url("https://ollama.example.com:443")


@pytest.mark.high_level
def test_security__localhost_ollama_endpoint_is_allowed_without_remote_allowlist():
    assert validate_ollama_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"


@pytest.mark.high_level
def test_security__query_length_boundary_rejects_oversized_input():
    with pytest.raises(ValueError, match="too long"):
        validate_query("x" * (MAX_QUERY_CHARS + 1))


@pytest.mark.high_level
def test_security__query_length_boundary_accepts_exact_limit():
    value = "x" * MAX_QUERY_CHARS
    assert validate_query(value) == value
