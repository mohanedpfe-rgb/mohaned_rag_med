from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_bootstrap_configures_page_before_auth():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    config_pos = source.index("st.set_page_config(")
    auth_pos = source.index("require_auth()")
    assert config_pos < auth_pos
    assert "bookrag_ui.st.set_page_config = lambda" in source


def test_no_public_default_admin_password():
    source = (ROOT / "rag_project" / "security.py").read_text(encoding="utf-8")
    assert "DEFAULT_LOCAL_PASSWORD" not in source
    assert "BOOKRAG_ADMIN_PASSWORD" in source


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
