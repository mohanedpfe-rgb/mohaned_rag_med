from __future__ import annotations

import hashlib

from rag_project.security import DEFAULT_ADMIN_PASSWORD_SHA256, MIN_ADMIN_PASSWORD_LENGTH, _configured_admin_passwords


def test_default_admin_password_digest_is_configured() -> None:
    digests = _configured_admin_passwords()
    assert DEFAULT_ADMIN_PASSWORD_SHA256 in digests
    assert len(DEFAULT_ADMIN_PASSWORD_SHA256) == 64


def test_default_digest_meets_password_length_policy() -> None:
    assert len(DEFAULT_ADMIN_PASSWORD_SHA256) == hashlib.sha256(b"x" * MIN_ADMIN_PASSWORD_LENGTH).digest_size * 2


def test_environment_password_adds_without_removing_default(monkeypatch) -> None:
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD", "a-valid-local-password-123")
    digests = _configured_admin_passwords()
    assert DEFAULT_ADMIN_PASSWORD_SHA256 in digests
    assert hashlib.sha256(b"a-valid-local-password-123").hexdigest() in digests
