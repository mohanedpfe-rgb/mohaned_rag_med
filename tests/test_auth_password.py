from __future__ import annotations

import hashlib

from rag_project.security import DEFAULT_ADMIN_PASSWORD_SHA256, _configured_admin_passwords


DEFAULT_PASSWORD = "becheikh_mohaned_rag"


def test_permanent_default_password_is_configured(monkeypatch):
    monkeypatch.delenv("BOOKRAG_ADMIN_PASSWORD", raising=False)
    expected = hashlib.sha256(DEFAULT_PASSWORD.encode("utf-8")).hexdigest()
    assert DEFAULT_ADMIN_PASSWORD_SHA256 == expected
    assert expected in _configured_admin_passwords()


def test_short_environment_password_does_not_remove_default(monkeypatch):
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD", "too-short")
    digests = _configured_admin_passwords()
    assert DEFAULT_ADMIN_PASSWORD_SHA256 in digests
    assert len(digests) == 1


def test_strong_environment_password_is_additional_to_default(monkeypatch):
    password = "a-valid-local-admin-password-123"
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD", password)
    digests = _configured_admin_passwords()
    assert DEFAULT_ADMIN_PASSWORD_SHA256 in digests
    assert hashlib.sha256(password.encode("utf-8")).hexdigest() in digests


def test_default_password_digest_is_sha256_length():
    assert len(DEFAULT_ADMIN_PASSWORD_SHA256) == 64
