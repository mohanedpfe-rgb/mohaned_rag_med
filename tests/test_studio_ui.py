from __future__ import annotations

from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "rag_project" / "app" / "studio_ui.py").read_text(encoding="utf-8")


def test_studio_ui_uses_composition_root_and_real_actions() -> None:
    source = _source()
    for marker in (
        "create_rag_system", "system.answer", "system.ingest_directory",
        "system.clear_pdf_data", "system.verify_index", "get_system.clear()", "st.rerun()",
    ):
        assert marker in source


def test_upload_deduplication_and_payload_validation_are_real() -> None:
    source = _source()
    for marker in (
        "hashlib.sha256", "saved_pdf_hashes", "if digest in seen",
        'content.startswith(b"%PDF-")', "incoming / f", "Path(name).name",
    ):
        assert marker in source


def test_canva_studio_information_architecture_is_complete() -> None:
    source = _source()
    for label in ("Overview", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"):
        assert f'"{label}"' in source
    for marker in (
        "BookRAG Studio", "Upload PDFs", "Incoming folder", "Start all chunks",
        "Recreate runtime", "Clear all PDF data", "Documents", "Ready", "Processing",
        "Vector chunks", "Lexical chunks", "Ollama", "Embedding", "Vector index",
        "Generation", "Search and answer", "Confidence", "Answerability", "Query quality",
        "Citations", "Evidence and trace", "Status", "File", "Stage", "Page", "Chunks",
        "Embeddings", "Dimension", "Error", "Page checkpoints", "Ollama host", "Chunk size",
        "Chunk overlap", "Top-k", "Temperature", "Vector weight", "Neighbor expansion",
    ):
        assert marker in source


def test_navigation_is_a_real_single_shell() -> None:
    source = _source()
    for marker in (
        "NAV_ITEMS", "def _navigate", 'st.session_state["studio_nav"]',
        "render_sidebar", "render_topbar", "render_overview", "render_chat",
        "render_health", "render_ingestion", "render_background", "render_inspector",
        "render_settings",
    ):
        assert marker in source


def test_chat_has_document_scope_and_grounding_metadata() -> None:
    source = _source()
    for marker in (
        "Search scope", "selected_filter", "metadata_filter", "system.answer",
        "console_answer", "studio_chat_nonce", "Citations", "evidence", "query_trace",
    ):
        assert marker in source
    assert 'key=f"studio_question_{nonce}"' in source
    assert 'st.session_state["studio_chat_nonce"] = nonce + 1' in source
    assert 'st.session_state["studio_question"] = ""' not in source


def test_health_inspector_and_ingestion_are_operational() -> None:
    source = _source()
    for marker in (
        "ollama_health", "Configured models missing from Ollama", "Recheck index",
        "verify_index", "Active progress", "Background workers",
        "Document and index inspector", "Runtime configuration",
    ):
        assert marker in source


def test_settings_and_ingestion_have_input_guards() -> None:
    source = _source()
    for marker in (
        "Chunk overlap must be smaller than chunk size.", "No PDF files were found",
        "Folder does not exist", "elif not ready", "path.suffix.lower() == \".pdf\"",
        "if not question.strip()",
    ):
        assert marker in source


def test_canva_geometry_contract_is_explicit() -> None:
    source = _source()
    for marker in (
        "max-width:1738px", "width:252px", "padding:23px 46px 56px",
        "grid-template-columns:minmax(0,1.36fr) minmax(330px,.84fr)",
        "min-height:202px", "border-radius:17px", "height:45px",
        "topbar-title", "hero-title", "metric-row", "content-grid",
    ):
        assert marker in source


def test_visual_system_matches_canva_style_language() -> None:
    source = _source()
    for marker in (
        "backdrop-filter", "metric-card", "panel", "status", "hero",
        "linear-gradient", "--surface", "--accent", "@media", "glass-note",
    ):
        assert marker in source
