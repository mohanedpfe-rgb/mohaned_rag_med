from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "rag_project/app/bookrag_ui.py").read_text(encoding="utf-8")


def test_ui_module_has_core_pages_and_runtime_calls():
    for marker in (
        "def home(",
        "def documents_page(",
        "def live_processing_page(",
        "def ask_page(",
        "def inspector_page(",
        "def system_page(",
        "def settings_page(",
        "system.answer",
        "system.ingest_directory",
        "system.verify_index",
        "system.health_report",
    ):
        assert marker in UI


def test_upload_path_validates_and_dispatches_processing():
    for marker in (
        "st.file_uploader",
        "accept_multiple_files=True",
        "hashlib.sha256",
        "validate_pdf_payload",
        "auto_ingest(system, added)",
        'trigger="upload"',
    ):
        assert marker in UI


def test_live_processing_uses_durable_state():
    for marker in (
        "state_store.get_pages",
        "state_store.get_events",
        "current_page",
        "total_pages",
        "ingestion_started_at",
        "ingestion_completed_at",
        "@st.fragment",
    ):
        assert marker in UI


def test_beginner_friendly_help_exists_without_legacy_wording_contract():
    for marker in ("Quick questions", "How to get stronger answers", "Source scope", "System health"):
        assert marker in UI


def test_ui_has_responsive_studio_css():
    for marker in ("block-container", "border-right", "linear-gradient", "box-shadow", "@media(max-width:1000px)"):
        assert marker in UI


def test_ui_uses_real_runtime_facade_helpers():
    assert "get_system()" in UI
    assert "render_live_runtime" in UI
    assert "require_auth()" in UI
