from __future__ import annotations

import importlib


def test_documented_default_password_is_always_accepted(monkeypatch):
    monkeypatch.delenv("BOOKRAG_ADMIN_PASSWORD", raising=False)
    security = importlib.import_module("rag_project.security")
    assert "becheikh_mohaned_rag" in security._configured_admin_passwords()


def test_documented_default_survives_stale_environment_override(monkeypatch):
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD", "some_old_or_stale_password")
    security = importlib.import_module("rag_project.security")
    passwords = security._configured_admin_passwords()
    assert "becheikh_mohaned_rag" in passwords
    assert "some_old_or_stale_password" in passwords
