from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_bootstrap_is_single_threaded_and_deterministic():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "bookrag_ui.main()" in source
    assert "threading.Thread" not in source
    assert "bookrag-runtime-bootstrap" not in source
    assert "bookrag_ui.st.set_page_config = lambda" not in source


def test_authentication_system_is_removed_from_runtime():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    security_source = (ROOT / "rag_project" / "security.py").read_text(encoding="utf-8")
    assert "require_auth" not in app_source
    assert "require_auth" not in security_source
    assert "BOOKRAG_ADMIN_PASSWORD" not in app_source
    assert "BOOKRAG_ADMIN_PASSWORD" not in security_source
    assert not (ROOT / "rag_project" / "auth.py").exists()


def test_supervisor_has_event_driven_and_reconciliation_guards():
    source = (ROOT / "rag_project" / "ingestion" / "auto_supervisor.py").read_text(encoding="utf-8")
    assert "Observer()" in source
    assert "_stable_enough" in source
    assert "_acquire_process_lock" in source
    assert "_trim_caches" in source


def test_runtime_transition_guard_is_installed():
    runtime = (ROOT / "rag_project" / "runtime.py").read_text(encoding="utf-8")
    guard = (ROOT / "rag_project" / "runtime_stability_v5.py").read_text(encoding="utf-8")
    assert "runtime_stability_v5" in runtime
    assert "Invalid terminal state regression" in guard


def test_streaming_client_requires_terminal_done_event():
    source = (ROOT / "rag_project" / "generation" / "llm_client.py").read_text(encoding="utf-8")
    assert "streaming response ended before the terminal done event" in source
