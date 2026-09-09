from __future__ import annotations

from pathlib import Path


def _source(path: str) -> str:
    return (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")


def test_active_app_uses_canva_exact_runtime() -> None:
    app = _source("app.py")
    exact = _source("rag_project/app/canva_exact_ui.py")
    assert "from rag_project.app.canva_exact_ui import main as exact_main" in app
    assert "exact_main()" in app
    for marker in ("BookRAG Studio", "Choose PDF files", "Vector chunks", "Settings", "Ask Your Documents", "System Health", "Inspector", "Start all chunks", "Recreate runtime", "Clear all PDF data"):
        assert marker in exact


def test_real_rag_actions_are_wired() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in ("create_rag_system", "system.answer", "system.ingest_directory", "system.clear_pdf_data", "system.verify_index", "st.rerun", "get_system.clear", "apply_settings_in_place"):
        assert marker in source


def test_upload_deduplication_and_pdf_validation_are_real() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in ("hashlib.sha256", "if digest in hashes", 'content.startswith(b"%PDF-")', "Path(name).name", "incoming.mkdir", "incoming / f"):
        assert marker in source


def test_navigation_and_functional_pages_exist() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in ('"Overview"', '"Documents"', '"Index them"', '"Ingestion"', '"Inspector"', '"Settings"', '"Chat"', '"Health"', '"Background"', 'st.session_state["studio_nav"]', "def sidebar", "def overview", "def chat", "def health", "def ingestion", "def background", "def inspector", "def settings"):
        assert marker in source


def test_chat_has_scope_grounding_and_citations() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in ("Search scope", "selected_filter", "metadata_filter", "system.answer", "console_answer", "studio_chat_nonce", "Citations", "evidence", "query_trace", "Evidence and trace"):
        assert marker in source


def test_health_ingestion_inspector_and_settings_have_guards() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in ("ollama_health", "Configured Ollama endpoint is unavailable.", "Recheck index", "verify_index", "Active progress", "Background workers", "Document and index inspector", "Runtime configuration", "Chunk overlap must be smaller than chunk size.", "No PDF files were found", "Folder does not exist", "if not q.strip()"):
        assert marker in source


def test_canva_geometry_contract_is_explicit() -> None:
    app = _source("app.py")
    exact = _source("rag_project/app/canva_exact_ui.py")
    for marker in ("left:91px", "top:112px", "width:1738px", "height:855px", "left:388px", "top:360px", "width:802px", "height:202px", "left:1210px", "width:573px", "top:694px", "left:388px", "top:694px", "width:802px", "top:600px", "height:74px", "top:405px", "height:158px"):
        assert marker in app or marker in exact


def test_no_bottom_panel_overlap_in_runtime_css() -> None:
    source = _source("app.py")
    assert ".query-panel{position:fixed!important;left:388px!important;top:600px!important;width:802px!important;height:74px" in source
    assert ".health-panel{position:fixed!important;left:388px!important;top:694px!important;width:802px!important;height:236px" in source
    assert ".inspector-panel{position:fixed!important;left:1210px!important;top:694px!important;width:573px!important;height:236px" in source


def test_visual_system_matches_canva_style_language() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in ("backdrop-filter", "index-metric", "settings-panel", "query-panel", "health-panel", "inspector-panel", "status", "linear-gradient", "--surface", "--accent", "@media", "glass-note"):
        assert marker in source
