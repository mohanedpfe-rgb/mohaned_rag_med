from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "rag_project/app/bookrag_ui.py").read_text(encoding="utf-8")


def test_ui_declares_all_primary_workspace_pages():
    for page in ("Home", "Documents", "Live Processing", "Ask BookRAG", "Inspector", "System", "Settings"):
        assert f'"{page}"' in UI
    for marker in ("def ask_page(", "def inspector_page(", "def system_page(", "def settings_page("):
        assert marker in UI


def test_upload_path_is_automatic_and_safe():
    for marker in (
        "st.file_uploader",
        "accept_multiple_files=True",
        "def save_pdf(",
        "auto_ingest(system",
        'trigger="upload"',
        "hashlib.sha256",
    ):
        assert marker in UI


def test_live_processing_reads_durable_state():
    for marker in (
        "state_store.get_pages",
        "state_store.get_events",
        "current_page",
        "total_pages",
        "ingestion_started_at",
        "ingestion_completed_at",
    ):
        assert marker in UI


def test_ui_has_beginner_friendly_research_controls():
    for marker in ("Quick questions", "How to get stronger answers", "Source scope", "System health"):
        assert marker in UI


def test_ui_has_responsive_studio_css():
    for marker in ("block-container", "border-right", "linear-gradient", "box-shadow"):
        assert marker in UI
    assert "@media" in UI


def test_ui_runtime_health_and_navigation_contract():
    for marker in ("get_system()", "health_report", "bookrag_page", "_navigate("):
        assert marker in UI
