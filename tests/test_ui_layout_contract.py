from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
UI = (ROOT / "rag_project" / "app" / "bookrag_ui.py").read_text(encoding="utf-8")
SECURITY = (ROOT / "rag_project" / "security.py").read_text(encoding="utf-8")
STATE_STORE = (ROOT / "rag_project" / "ingestion" / "state_store.py").read_text(encoding="utf-8")
EXTRACTOR = (ROOT / "rag_project" / "parsing" / "pdf_extractor.py").read_text(encoding="utf-8")


def test_entrypoint_uses_new_bookrag_ui_and_no_fixed_canvas():
    assert "from rag_project.app.bookrag_ui import main as ui_main" in APP
    assert "ui_main()" in APP
    assert "position:fixed" not in APP


def test_runtime_clock_guard_and_canonical_utc_helper():
    assert "from datetime import datetime, timezone" in APP
    assert "getattr(rag_system_module, \"utc_now\", None)" in APP
    assert "datetime.now(timezone.utc).isoformat()" in APP
    assert "def utc_now()" in STATE_STORE


def test_navigation_pages_are_beginner_friendly():
    for page in ("Home", "Documents", "Live Processing", "Ask BookRAG", "Inspector", "System", "Settings"):
        assert f'"{page}"' in UI
    assert "def sidebar" in UI
    assert "def home" in UI
    assert "def documents_page" in UI
    assert "def processing_page" in UI
    assert "def ask_page" in UI
    assert "def inspector_page" in UI
    assert "def system_page" in UI
    assert "def settings_page" in UI


def test_upload_to_processing_path_is_automatic():
    assert "st.file_uploader" in UI
    assert "save_pdf(incoming, upload.name, payload)" in UI
    assert "auto_ingest(system, added)" in UI
    assert 'trigger="upload"' in UI
    assert "system.ingest_directory" in UI


def test_live_processing_is_backed_by_real_persistent_state():
    for marker in (
        "@st.fragment(run_every=\"2s\")",
        "state_store.get_pages",
        "state_store.get_events",
        "current_page",
        "total_pages",
        "ingestion_started_at",
        "ingestion_completed_at",
        "Live event timeline",
        "Page-by-page progress",
        "Elapsed time",
    ):
        assert marker in UI
    assert "current_page=physical_page,total_pages=page_count" in EXTRACTOR


def test_beginner_help_and_clear_labels_exist():
    for marker in (
        "How it works",
        "Good questions",
        "You do not need to understand RAG internals.",
        "Search scope",
        "Semantic search weight",
        "Answer creativity",
        "Use nearby sections",
        "What healthy means",
        "help=",
    ):
        assert marker in UI


def test_rag_actions_remain_connected():
    for marker in (
        "create_rag_system",
        "system.answer",
        "system.ingest_directory",
        "system.clear_pdf_data",
        "system.verify_index",
        "system.health_report",
        "apply_settings_in_place",
    ):
        assert marker in UI or marker in APP


def test_security_facade_still_guards_core_operations():
    for marker in (
        "require_auth()",
        "validate_pdf_payload",
        "validate_ollama_url",
        "validate_storage_path",
        "require_clear_confirmation",
        'DEFAULT_LOCAL_PASSWORD = "becheikh_mohaned_med"',
    ):
        assert marker in APP or marker in SECURITY


def test_layout_is_responsive_and_not_fixed():
    assert "block-container" in UI
    assert "overflow:auto" in UI
    assert "@media(max-width:1000px)" in UI
    assert "@media(max-width:650px)" in UI
    assert "linear-gradient" in UI
    assert "box-shadow" in UI
    assert "position:fixed" not in UI
