from pathlib import Path

import fitz
import pytest

from rag_project.ingestion.state_store import IngestionStateStore
from rag_project.security import (
    CLEAR_PHRASE,
    MAX_SESSION_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    max_pdf_pages,
    register_session_upload,
    sanitize_log_text,
    sanitize_model_text,
    validate_ollama_url,
    validate_pdf_page_count,
    validate_pdf_payload,
    validate_query,
    validate_storage_path,
)


def _valid_pdf_bytes() -> bytes:
    doc = fitz.open()
    doc.new_page()
    payload = doc.tobytes()
    doc.close()
    return payload


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


def test_remote_ollama_is_denied_without_exact_allowlist(monkeypatch):
    monkeypatch.delenv("BOOKRAG_OLLAMA_ALLOWLIST", raising=False)
    with pytest.raises(ValueError):
        validate_ollama_url("http://example.com:11434")


def test_ollama_allowlist_rejects_private_dns(monkeypatch):
    import rag_project.security as security
    monkeypatch.setenv("BOOKRAG_OLLAMA_ALLOWLIST", "ollama.example")
    monkeypatch.setattr(security, "_resolved_ips", lambda host: {"10.0.0.4"})
    with pytest.raises(ValueError):
        validate_ollama_url("https://ollama.example:11434")


def test_ollama_credentials_and_extra_url_parts_are_rejected():
    with pytest.raises(ValueError):
        validate_ollama_url("http://user:password@127.0.0.1:11434")
    with pytest.raises(ValueError):
        validate_ollama_url("http://127.0.0.1:11434/api")
    with pytest.raises(ValueError):
        validate_ollama_url("http://127.0.0.1:11434?x=1")


def test_pdf_validation_checks_structure_size_and_page_limit():
    payload = _valid_pdf_bytes()
    validate_pdf_payload("ok.pdf", payload)
    with pytest.raises(ValueError):
        validate_pdf_payload("bad.pdf", b"MZ" + b"0" * 1024)
    with pytest.raises(ValueError):
        validate_pdf_payload("fake.pdf", b"%PDF-1.7\n")
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
    store.upsert_document({
        "document_id": document_id,
        "content_hash": "hash-1",
        "file_path": str(tmp_path / "incoming" / "a.pdf"),
        "file_name": "a.pdf",
        "file_size": 1,
        "parser_version": "test",
        "ocr_config": "{}",
        "chunking_config": "{}",
        "embedding_model": "test",
    })
    with pytest.raises(ValueError):
        store.update_document(document_id, **{"status = 'READY', error": "bad"})


def test_untrusted_text_and_logs_are_bounded():
    assert "[TRUNCATED_UNTRUSTED_TEXT]" in sanitize_model_text("x" * 20, limit=10)
    cleaned = sanitize_log_text("hello\nforged log\r\n")
    assert "\\n" in cleaned and "\\r" in cleaned
