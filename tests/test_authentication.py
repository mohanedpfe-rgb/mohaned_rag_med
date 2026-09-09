from __future__ import annotations

from rag_project.auth import _verify_password, password_hash


def test_scrypt_password_verifier_round_trip(monkeypatch) -> None:
    verifier = password_hash("correct horse battery staple", salt=b"0123456789abcdef")
    monkeypatch.delenv("BOOKRAG_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD_HASH", verifier)

    assert _verify_password("correct horse battery staple") is True
    assert _verify_password("incorrect password") is False


def test_plain_password_verification_is_constant_time(monkeypatch) -> None:
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD", "strong-password")
    monkeypatch.delenv("BOOKRAG_ADMIN_PASSWORD_HASH", raising=False)

    assert _verify_password("strong-password") is True
    assert _verify_password("wrong-password") is False
