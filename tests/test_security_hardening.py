from pathlib import Path

import pytest

from rag_project.security import (
    CLEAR_PHRASE,
    MAX_UPLOAD_BYTES,
    validate_ollama_url,
    validate_pdf_payload,
    validate_query,
    validate_storage_path,
)


def test_storage_path_is_jailed(tmp_path: Path):
    root = tmp_path / "bookrag"
    root.mkdir()
    assert validate_storage_path(root, root / "data" / "incoming").is_relative_to(root)
    with pytest.raises(ValueError):
        validate_storage_path(root, tmp_path / "outside")
    with pytest.raises(ValueError):
        validate_storage_path(root, root / "data" / ".." / ".." / "escape")


def test_ollama_defaults_to_loopback():
    assert validate_ollama_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert validate_ollama_url("http://localhost:11434") == "http://localhost:11434"


def test_remote_ollama_is_denied_by_default(monkeypatch):
    monkeypatch.delenv("BOOKRAG_ALLOW_REMOTE_OLLAMA", raising=False)
    monkeypatch.delenv("BOOKRAG_OLLAMA_ALLOWLIST", raising=False)
    with pytest.raises(ValueError):
        validate_ollama_url("http://192.168.1.20:11434")


def test_ollama_credentials_are_never_accepted():
    with pytest.raises(ValueError):
        validate_ollama_url("http://user:password@127.0.0.1:11434")


def test_pdf_size_and_magic_byte_limits():
    validate_pdf_payload("ok.pdf", b"%PDF-1.7\n")
    with pytest.raises(ValueError):
        validate_pdf_payload("bad.pdf", b"MZ" + b"0" * 1024)
    with pytest.raises(ValueError):
        validate_pdf_payload("large.pdf", b"%PDF-" + b"0" * MAX_UPLOAD_BYTES)


def test_query_length_limit():
    assert validate_query(" hello ") == "hello"
    with pytest.raises(ValueError):
        validate_query("x" * 4001)


def test_clear_phrase_is_explicit():
    assert CLEAR_PHRASE == "CLEAR ALL PDF DATA"
