from __future__ import annotations

from pathlib import Path


def _source(path: str) -> str:
    return (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")


def test_active_app_uses_new_bookrag_ui() -> None:
    app = _source("app.py")
    ui = _source("rag_project/app/bookrag_ui.py")
    assert "from rag_project.app.bookrag_ui import main as ui_main" in app
    assert "ui_main()" in app
    for marker in ("BookRAG", "Home", "Documents", "Live Processing", "Ask BookRAG", "Inspector", "System", "Settings"):
        assert marker in ui


def test_ui_is_functional_and_uses_real_rag_runtime() -> None:
    source = _source("rag_project/app/bookrag_ui.py")
    for marker in (
        "create_rag_system",
        "system.answer",
        "system.ingest_directory",
        "system.clear_pdf_data",
        "system.verify_index",
        "system.health_report",
        "apply_settings_in_place",
        "@st.fragment(run_every=\"2s\")",
    ):
        assert marker in source


def test_upload_starts_processing_automatically() -> None:
    source = _source("rag_project/app/bookrag_ui.py")
    for marker in (
        'accept_multiple_files=True',
        "hashlib.sha256",
        "if digest in saved",
        'content.startswith(b"%PDF-")',
        'trigger="upload"',
        "Automatic processing is now running.",
        "auto_ingest(system, added)",
    ):
        assert marker in source


def test_exact_page_progress_is_persisted_every_page() -> None:
    source = _source("rag_project/parsing/pdf_extractor.py")
    assert "current_page=physical_page,total_pages=page_count" in source
    assert "upsert_page(document_id,physical_page" in source


def test_live_page_inspection_uses_persisted_pages_and_events() -> None:
    source = _source("rag_project/app/bookrag_ui.py")
    for marker in (
        "state_store.get_pages",
        "state_store.get_events",
        "Page-by-page progress",
        "Current page",
        "Elapsed time",
        "Live event timeline",
        "updates every 2s",
    ):
        assert marker in source


def test_beginner_friendly_labels_and_help_exist() -> None:
    source = _source("rag_project/app/bookrag_ui.py")
    for marker in (
        "How it works",
        "Good questions",
        "Search scope",
        "Semantic search weight",
        "Answer creativity",
        "Use nearby sections",
        "What healthy means",
        "You do not need to understand RAG internals.",
    ):
        assert marker in source


def test_professional_sidebar_and_responsive_layout() -> None:
    source = _source("rag_project/app/bookrag_ui.py")
    for marker in (
        "[data-testid=\"stSidebar\"]",
        "border-right:1px solid",
        "linear-gradient",
        "box-shadow",
        "@media(max-width:1000px)",
        "@media(max-width:650px)",
        "overflow:auto",
    ):
        assert marker in source
    assert "position:fixed" not in source


def test_utc_guard_remains_in_entrypoint() -> None:
    app = _source("app.py")
    state_store = _source("rag_project/ingestion/state_store.py")
    assert "datetime.now(timezone.utc).isoformat()" in app
    assert "def utc_now()" in state_store
