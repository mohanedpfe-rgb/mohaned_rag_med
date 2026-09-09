from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
UI = (ROOT / "rag_project" / "app" / "canva_exact_ui.py").read_text(encoding="utf-8")
SECURITY = (ROOT / "rag_project" / "security.py").read_text(encoding="utf-8")
STATE_STORE = (ROOT / "rag_project" / "ingestion" / "state_store.py").read_text(encoding="utf-8")


def test_entrypoint_is_thin_and_has_no_legacy_layout_css():
    assert "exact_main()" in APP
    assert "clear_confirmation_ui" not in APP
    assert "position:fixed" not in APP
    assert "width:1441px" not in APP
    assert "height:855px" not in APP


def test_runtime_clock_guard_prevents_undefined_utc_now_failures():
    assert "from datetime import datetime, timezone" in APP
    assert "rag_system_module" in APP
    assert "getattr(rag_system_module, \"utc_now\", None)" in APP
    assert "datetime.now(timezone.utc).isoformat()" in APP
    assert "def utc_now()" in STATE_STORE


def test_ui_has_real_navigation_and_dispatch_for_every_page():
    assert "st.session_state[\"studio_nav\"]" in UI
    assert "def _navigate(page: str)" in UI
    for page in ("Overview", "Documents", "Index them", "Ingestion", "Chat", "Inspector", "Health", "Settings", "Background"):
        assert f'page == "{page}"' in UI or f'["{page}"]' in UI
    assert 'key="nav_Overview"' in UI
    assert 'key="top_chat"' in UI
    assert 'key="top_health"' in UI
    assert 'key="top_refresh"' in UI


def test_shared_runtime_and_worker_state_are_explicit():
    assert "@st.cache_resource(show_spinner=False)" in UI
    assert "def get_system" in UI
    assert "def get_jobs" in UI
    assert "registry = get_jobs()" in UI
    assert "system.ingest_directory" in UI
    assert "system.answer" in UI


def test_upload_is_the_primary_automatic_ingestion_trigger():
    assert "accept_multiple_files=True" in UI
    assert "save_pdf(incoming, upload.name, payload)" in UI
    assert "def auto_start_after_upload" in UI
    assert 'trigger="upload"' in UI
    assert 'st.session_state["studio_last_job"] = job_id' in UI
    assert 'st.session_state["studio_nav"] = "Ingestion"' in UI
    assert "Retry waiting PDFs" in UI
    assert "Indexing starts automatically" in UI


def test_ingestion_worker_has_shared_lifecycle_and_results():
    assert "trigger" in UI
    assert "_set_job_state(job" in UI
    assert "status=state" in UI
    assert "completed" in UI and "failed" in UI
    assert "finished=time.time()" in UI


def test_live_monitoring_is_real_and_not_a_fake_fixed_progress_bar():
    assert "@st.fragment(run_every=\"2s\")" in UI
    assert "@st.fragment(run_every=\"3s\")" in UI
    assert "_document_progress" in UI
    assert "_latest_events_by_document" in UI
    assert "state_store.get_events" in UI
    assert "Estimated pipeline progress" in UI


def test_beginner_friendly_tooling_is_explained():
    assert "How to use BookRAG in 3 steps" in UI
    assert "help=" in UI
    assert "What does each stage mean?" in UI
    assert "Overall readiness" in UI
    assert "Manual recovery" in UI


def test_workflow_is_connected_documents_chat_inspector_health():
    assert "ready_docs(system)" in UI
    assert "metadata_filter" in UI
    assert "system.answer" in UI
    assert "system.verify_index" in UI
    assert "system.health_report" in UI
    assert "ollama_health(system.settings.ollama_base_url)" in UI


def test_security_facade_still_guards_core_operations():
    assert "require_auth()" in APP
    assert "validate_pdf_payload" in APP
    assert "validate_ollama_url" in APP
    assert "validate_storage_path" in APP
    assert "require_clear_confirmation" in APP
    assert 'DEFAULT_LOCAL_PASSWORD = "becheikh_mohaned_med"' in SECURITY


def test_layout_uses_normal_flow_and_contained_tables():
    assert "block-container" in UI
    assert "overflow:auto" in UI
    assert "min-width:0" in UI
    assert "grid-template-columns" in UI
    assert "position:fixed" not in UI
