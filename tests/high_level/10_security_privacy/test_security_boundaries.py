from __future__ import annotations

from pathlib import Path

import pytest

from rag_project.security import sanitize_log_text, sanitize_model_text, validate_pdf_payload, validate_storage_path


@pytest.mark.high_level
def test_security__path_traversal_is_rejected(tmp_path):
    root = tmp_path / "bookrag"
    root.mkdir()
    with pytest.raises(ValueError, match="inside the BookRAG project directory"):
        validate_storage_path(root, root / ".." / "outside.pdf", label="candidate")


@pytest.mark.high_level
def test_security__production_ingestion_rejects_non_pdf(clean_system, tmp_path):
    candidate = tmp_path / "notes.txt"
    candidate.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported or missing PDF"):
        clean_system.ingest_file(candidate)


@pytest.mark.high_level
def test_security__production_ingestion_rejects_path_escape(clean_system, tmp_path):
    candidate = tmp_path / "escape.pdf"
    candidate.write_bytes(b"%PDF-1.4\n% deliberately outside controlled project roots\n")
    with pytest.raises(ValueError, match="outside|inside|Unsupported|missing"):
        clean_system.ingest_file(candidate)


@pytest.mark.high_level
def test_security__non_pdf_upload_is_rejected():
    with pytest.raises(ValueError, match="not a valid PDF payload"):
        validate_pdf_payload("notes.txt", b"plain text")


@pytest.mark.high_level
def test_security__control_characters_are_removed_from_logged_and_model_text():
    raw = "patient\x00name\x1b[31msecret\x7f"
    sanitized_log = sanitize_log_text(raw)
    sanitized_model = sanitize_model_text(raw, limit=100)
    assert "\x00" not in sanitized_log
    assert "\x1b" not in sanitized_model
    assert "\x7f" not in sanitized_model


@pytest.mark.high_level
def test_security__oversized_model_text_is_explicitly_truncated():
    limit = 20
    result = sanitize_model_text("x" * 100, limit=limit)
    marker = "[TRUNCATED_UNTRUSTED_TEXT]"
    assert marker in result
    assert result.startswith("x" * limit)
    assert len(result) == limit + 1 + len(marker)
