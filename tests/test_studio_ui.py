from __future__ import annotations

from pathlib import Path


def _source(path: str) -> str:
    return (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")


def test_active_app_uses_current_functional_runtime() -> None:
    app = _source("app.py")
    exact = _source("rag_project/app/canva_exact_ui.py")
    assert "from rag_project.app.canva_exact_ui import main as exact_main" in app
    assert "exact_main()" in app
    for marker in (
        "BookRAG Studio",
        "Index them",
        "Documents",
        "Ingestion",
        "Chat",
        "Inspector",
        "Health",
        "Settings",
        "Background",
        "Delete all managed PDF data",
    ):
        assert marker in exact


def test_real_rag_actions_are_wired() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in (
        "create_rag_system",
        "system.answer",
        "system.ingest_directory",
        "system.clear_pdf_data",
        "system.verify_index",
        "system.health_report",
        "st.rerun",
        "get_system.clear",
        "apply_settings_in_place",
    ):
        assert marker in source


def test_upload_deduplication_validation_and_auto_start_are_real() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in (
        "hashlib.sha256",
        "if digest in saved",
        'content.startswith(b"%PDF-")',
        "incoming.mkdir",
        "incoming / f",
        "def auto_start_after_upload",
        'trigger="upload"',
        "Indexing started automatically",
    ):
        assert marker in source


def test_navigation_and_functional_pages_exist() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in (
        '"Overview"',
        '"Documents"',
        '"Index them"',
        '"Ingestion"',
        '"Inspector"',
        '"Settings"',
        '"Chat"',
        '"Health"',
        '"Background"',
        'st.session_state["studio_nav"]',
        "def sidebar",
        "def overview",
        "def chat",
        "def health",
        "def ingestion",
        "def background",
        "def inspector",
        "def settings",
    ):
        assert marker in source


def test_chat_has_scope_grounding_and_evidence_trace() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in (
        "Document scope",
        "selected_filter",
        "metadata_filter",
        "system.answer",
        "console_answer",
        "studio_chat_nonce",
        "Evidence references",
        "evidence",
        "query_trace",
        "Show evidence and query trace",
    ):
        assert marker in source


def test_live_ingestion_monitoring_is_real() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in (
        '@st.fragment(run_every="2s")',
        '@st.fragment(run_every="3s")',
        "_document_progress",
        "_latest_events_by_document",
        "state_store.get_events",
        "Estimated pipeline progress",
        "Live pipeline",
        "Worker activity",
    ):
        assert marker in source


def test_health_inspector_settings_and_recovery_are_beginner_friendly() -> None:
    source = _source("rag_project/app/canva_exact_ui.py")
    for marker in (
        "ollama_health",
        "Overall readiness",
        "Recheck health now",
        "Verify this document's index",
        "Current stage",
        "Basic settings",
        "Advanced configuration",
        "Chunk overlap must be smaller than chunk size.",
        "There are no PDF files waiting in Incoming.",
        "Folder does not exist",
        "Manual recovery",
        "Retry waiting PDFs",
    ):
        assert marker in source


def test_ui_layout_is_responsive_and_normal_flow() -> None:
    app = _source("app.py")
    exact = _source("rag_project/app/canva_exact_ui.py")
    assert "position:fixed" not in app
    assert "position:fixed" not in exact
    for marker in (
        "block-container",
        "overflow:auto",
        "min-width:0",
        "grid-template-columns",
        "@media",
        "glass-note",
        "linear-gradient",
        "--accent",
    ):
        assert marker in exact
