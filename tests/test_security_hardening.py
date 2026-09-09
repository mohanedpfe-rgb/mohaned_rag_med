from pathlib import Path

import pytest

from rag_project.security import (
    CLEAR_PHRASE,
    MAX_SESSION_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    max_pdf_pages,
    register_session_upload,
    validate_ollama_url,
    validate_pdf_page_count,
    validate_pdf_payload,
    validate_query,
    validate_storage_path,
)
from rag_project.ingestion.state_store import IngestionStateStore


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


def test_pdf_page_limit(monkeypatch):
    monkeypatch.delenv("BOOKRAG_MAX_PDF_PAGES", raising=False)
    assert max_pdf_pages() == 500
    assert validate_pdf_page_count(500) == 500
    with pytest.raises(ValueError):
        validate_pdf_page_count(501)


def test_pdf_page_limit_can_be_tuned(monkeypatch):
    monkeypatch.setenv("BOOKRAG_MAX_PDF_PAGES", "1000")
    assert max_pdf_pages() == 1000
    assert validate_pdf_page_count(1000) == 1000
    with pytest.raises(ValueError):
        validate_pdf_page_count(1001)
    monkeypatch.setenv("BOOKRAG_MAX_PDF_PAGES", "99999")
    assert max_pdf_pages() == 5000


def test_query_length_limit():
    assert validate_query(" hello ") == "hello"
    with pytest.raises(ValueError):
        validate_query("x" * 4001)


def test_clear_phrase_is_explicit():
    assert CLEAR_PHRASE == "CLEAR ALL PDF DATA"


def test_register_session_upload_limit(monkeypatch):
    import rag_project.security as security

    session = {}
    monkeypatch.setattr(security.st, "session_state", session)
    register_session_upload(MAX_SESSION_UPLOAD_BYTES)
    assert session["bookrag_upload_bytes"] == MAX_SESSION_UPLOAD_BYTES
    with pytest.raises(ValueError):
        register_session_upload(1)


def test_document_update_rejects_untrusted_sql_identifiers(tmp_path: Path):
    store = IngestionStateStore(tmp_path / "state.sqlite3")
    document_id = "doc-1"
    store.upsert_document(
        {
            "document_id": document_id,
            "content_hash": "hash-1",
            "file_path": str(tmp_path / "incoming" / "a.pdf"),
            "file_name": "a.pdf",
            "file_size": 1,
            "parser_version": "test",
            "ocr_config": "{}",
            "chunking_config": "{}",
            "embedding_model": "test",
        }
    )
    with pytest.raises(ValueError):
        store.update_document(document_id, **{"status = 'READY', error": "bad"})
