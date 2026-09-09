from __future__ import annotations

import importlib


def test_admin_password_is_rejected_when_missing(monkeypatch):
    monkeypatch.delenv("BOOKRAG_ADMIN_PASSWORD", raising=False)
    security = importlib.import_module("rag_project.security")
    assert security._configured_admin_passwords() == ()


def test_admin_password_requires_minimum_length(monkeypatch):
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD", "too-short")
    security = importlib.import_module("rag_project.security")
    assert security._configured_admin_passwords() == ()


def test_configured_strong_password_is_accepted(monkeypatch):
    password = "a-strong-local-admin-password"
    monkeypatch.setenv("BOOKRAG_ADMIN_PASSWORD", password)
    security = importlib.import_module("rag_project.security")
    assert security._configured_admin_passwords() == (password,)


def test_no_implicit_default_password_exists(monkeypatch):
    monkeypatch.delenv("BOOKRAG_ADMIN_PASSWORD", raising=False)
    security = importlib.import_module("rag_project.security")
    assert not hasattr(security, "DEFAULT_ADMIN_PASSWORD")
