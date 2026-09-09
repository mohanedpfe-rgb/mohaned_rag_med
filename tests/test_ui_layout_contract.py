from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "rag_project/app/bookrag_ui.py").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
SECURITY = (ROOT / "rag_project/security.py").read_text(encoding="utf-8")


def test_upload_to_processing_contract_is_automatic():
    assert "st.file_uploader" in UI
    assert "validate_pdf_payload" in APP or "validate_pdf_payload" in UI
    assert "auto_ingest(system" in UI
    assert 'trigger="upload"' in UI


def test_live_processing_is_backed_by_persistent_state():
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


def test_security_facade_guards_core_operations():
    for marker in (
        "require_auth()",
        "validate_pdf_payload",
        "validate_ollama_url",
        "validate_storage_path",
        "require_clear_confirmation",
        "DEFAULT_ADMIN_PASSWORD",
    ):
        assert marker in APP or marker in SECURITY


def test_layout_is_responsive():
    assert "block-container" in UI
    assert "@media(max-width:1000px)" in UI
    assert "@media(max-width:650px)" in UI


def test_main_theme_and_studio_sections_are_present():
    for marker in (".stApp", "linear-gradient", "box-shadow", "System health", "BookRAG"):
        assert marker in UI
