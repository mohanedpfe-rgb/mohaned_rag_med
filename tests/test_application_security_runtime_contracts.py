from __future__ import annotations

from types import SimpleNamespace

import pytest

import rag_project.application as application
import rag_project.security as security


def test_create_rag_system_returns_nonready_system_when_quality_gate_raises(monkeypatch):
    calls = []

    class DummySystem:
        def __init__(self, settings):
            self.settings = settings
            self.logger = SimpleNamespace(
                warning=lambda *args, **kwargs: calls.append(("warning", args)),
                exception=lambda *args, **kwargs: calls.append(("exception", args)),
            )
            self.startup_quality = None

    monkeypatch.setattr(application, "install", lambda: calls.append(("install",)))
    monkeypatch.setattr(application, "harden_system", lambda system: system)
    monkeypatch.setattr(application, "run_quality_gate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    import rag_project.app.production_rag as production_rag

    monkeypatch.setattr(production_rag, "ProductionRAGSystem", DummySystem)
    settings = SimpleNamespace(
        embedding_batch_size=16,
        embedding_retries=1,
        embedding_timeout_seconds=30,
        max_workers=1,
        ollama_concurrency=1,
    )
    result = application.create_rag_system(settings)
    assert result.startup_quality == {"ready": False, "error": "RuntimeError"}
    assert calls[0][0] == "install"
    assert any(item[0] == "exception" for item in calls)


def test_create_rag_system_warns_when_quality_gate_returns_not_ready(monkeypatch):
    calls = []

    class DummySystem:
        def __init__(self, settings):
            self.settings = settings
            self.logger = SimpleNamespace(
                warning=lambda *args, **kwargs: calls.append(("warning", args)),
                exception=lambda *args, **kwargs: calls.append(("exception", args)),
            )
            self.startup_quality = None

    monkeypatch.setattr(application, "install", lambda: None)
    monkeypatch.setattr(application, "harden_system", lambda system: system)
    monkeypatch.setattr(application, "run_quality_gate", lambda *args, **kwargs: {"ready": False, "reason": "drift"})
    import rag_project.app.production_rag as production_rag

    monkeypatch.setattr(production_rag, "ProductionRAGSystem", DummySystem)
    settings = SimpleNamespace(
        embedding_batch_size=16,
        embedding_retries=1,
        embedding_timeout_seconds=30,
        max_workers=1,
        ollama_concurrency=1,
    )
    result = application.create_rag_system(settings)
    assert result.startup_quality["ready"] is False
    assert any(item[0] == "warning" for item in calls)


def test_runtime_settings_normalization_never_allows_zero_or_subminimum(monkeypatch):
    settings = SimpleNamespace(
        embedding_batch_size=0,
        embedding_retries=-100,
        embedding_timeout_seconds=0,
        max_workers=0,
        ollama_concurrency=0,
    )
    normalized = application._normalize_runtime_settings(settings)
    assert normalized.embedding_batch_size == 16
    assert normalized.embedding_retries == 1
    assert normalized.embedding_timeout_seconds == 30.0
    assert normalized.max_workers == 1
    assert normalized.ollama_concurrency == 1


def test_require_clear_confirmation_fails_without_exact_phrase(monkeypatch):
    monkeypatch.setitem(security.st.session_state, "bookrag_clear_phrase", "clear all pdf data")
    with pytest.raises(PermissionError):
        security.require_clear_confirmation()


def test_require_clear_confirmation_accepts_exact_phrase_and_audits(monkeypatch):
    monkeypatch.setitem(security.st.session_state, "bookrag_clear_phrase", security.CLEAR_PHRASE)
    events = []
    monkeypatch.setattr(security, "audit_event", lambda action, **kwargs: events.append(action))
    security.require_clear_confirmation()
    assert events == ["clear_authorized"]


def test_consume_rate_limit_enforces_limit_and_resets_after_window(monkeypatch):
    monkeypatch.setattr(security, "_session_actor", lambda: "test-actor-rate")
    security._RATE_STATE.clear()
    clock = {"now": 100.0}
    monkeypatch.setattr(security.time, "monotonic", lambda: clock["now"])
    assert security.consume_rate_limit("unit", limit=2, window_seconds=10) is True
    assert security.consume_rate_limit("unit", limit=2, window_seconds=10) is True
    assert security.consume_rate_limit("unit", limit=2, window_seconds=10) is False
    clock["now"] = 111.0
    assert security.consume_rate_limit("unit", limit=2, window_seconds=10) is True
    security._RATE_STATE.clear()


def test_session_upload_counter_rejects_negative_and_cumulative_overflow(monkeypatch):
    monkeypatch.setitem(security.st.session_state, "bookrag_upload_bytes", 0)
    with pytest.raises(ValueError, match="negative"):
        security.register_session_upload(-1)
    monkeypatch.setattr(security, "MAX_SESSION_UPLOAD_BYTES", 10)
    security.register_session_upload(6)
    assert security.st.session_state["bookrag_upload_bytes"] == 6
    with pytest.raises(ValueError, match="500 MB cumulative"):
        security.register_session_upload(5)


def test_slot_guards_can_acquire_and_release_without_leaking(monkeypatch):
    monkeypatch.setattr(security, "_session_actor", lambda: "slot-test")
    assert security.acquire_ingest_slot(0) is True
    security.release_ingest_slot()
    assert security.acquire_answer_slot(0) is True
    security.release_answer_slot()


def test_security_path_validation_rejects_file_named_like_directory(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    file_path = root / "existing.txt"
    file_path.write_text("x", encoding="utf-8")
    resolved = security.validate_storage_path(root, file_path, "path")
    assert resolved == file_path.resolve()


def test_sanitize_evidence_handles_empty_and_control_only_inputs():
    assert security.sanitize_evidence_for_prompt("") == ""
    assert security.sanitize_evidence_for_prompt("\x00\x01\x02") == ""


def test_max_pdf_pages_never_becomes_zero(monkeypatch):
    monkeypatch.setenv("BOOKRAG_MAX_PDF_PAGES", "0")
    assert security.max_pdf_pages() == 1
    monkeypatch.setenv("BOOKRAG_MAX_PDF_PAGES", "-5")
    assert security.max_pdf_pages() == 1
