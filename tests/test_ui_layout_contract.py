from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "rag_project/app/bookrag_ui.py").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
SECURITY = (ROOT / "rag_project/security.py").read_text(encoding="utf-8")


def test_upload_to_processing_contract_is_automatic():
    for marker in ("st.file_uploader", "accept_multiple_files=True", "def save_pdf(", "auto_ingest(system", 'trigger="upload"'):
        assert marker in UI
    assert "validate_pdf_payload" in APP or "validate_pdf_payload" in SECURITY


def test_live_processing_is_backed_by_persistent_state():
    for marker in ("state_store.get_pages", "state_store.get_events", "current_page", "total_pages", "ingestion_started_at", "ingestion_completed_at"):
        assert marker in UI


def test_security_facade_guards_core_operations():
    for marker in ("validate_pdf_payload", "validate_ollama_url", "validate_storage_path", "require_clear_confirmation"):
        assert marker in APP or marker in SECURITY
    assert "require_auth" not in APP
    assert "require_auth" not in SECURITY
    assert "DEFAULT_ADMIN_PASSWORD" not in APP
    assert "DEFAULT_ADMIN_PASSWORD" not in SECURITY


def test_layout_is_responsive_or_uses_streamlit_native_layout():
    assert "block-container" in UI
    assert "linear-gradient" in UI
    assert "box-shadow" in UI
    assert "st.columns" in UI
    assert "@media" in UI or "use_container_width" in UI


def test_main_theme_and_studio_sections_are_present():
    for marker in (".stApp", "System health", "BookRAG", "PAGES ="):
        assert marker in UI
