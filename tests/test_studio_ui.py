from __future__ import annotations

from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "rag_project" / "app" / "studio_ui.py").read_text(encoding="utf-8")


def test_studio_ui_uses_composition_root_and_real_actions() -> None:
    source = _source()

    assert "create_rag_system" in source
    assert "system.answer" in source
    assert "system.ingest_directory" in source
    assert "system.clear_pdf_data" in source
    assert "system.verify_index" in source
    assert "system.vector_store.compatibility_report" in source
    assert "st.rerun()" in source


def test_upload_deduplication_and_payload_validation_are_real() -> None:
    source = _source()

    assert "hashlib.sha256" in source
    assert "saved_pdf_hashes" in source
    assert "if digest in seen" in source
    assert 'content.startswith(b"%PDF-")' in source
    assert "incoming / f" in source


def test_canva_studio_information_architecture_is_present() -> None:
    source = _source()

    # The Canva concept is a single Studio shell with these primary destinations.
    for label in ("Overview", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"):
        assert f'"{label}"' in source
    assert "BookRAG Studio" in source
    assert "Start all chunks" in source
    assert "system.apply_settings_in_place" in source
    assert "console_answer" in source
    assert "Page checkpoints" in source
    assert "Recent process events" in source


def test_canva_redesign_has_complete_functional_paths() -> None:
    source = _source()

    assert "NAV_ITEMS" in source
    assert "def _navigate" in source
    assert "metadata_filter={\"document_id\": selected_filter}" in source
    assert "Search scope" in source
    assert "Ask about this document" in source
    assert "Recheck index" in source
    assert "Apply live settings" in source
    assert "Configured models missing from Ollama" in source
    assert "Active progress" in source
    assert "Latest ingestion completed" in source or "Latest ingestion failed" in source


def test_chat_clear_is_streamlit_safe() -> None:
    source = _source()

    # Clearing chat rotates the widget key instead of mutating a live widget's
    # session-state value after Streamlit has instantiated it.
    assert "studio_chat_nonce" in source
    assert 'key=f"studio_question_{chat_nonce}"' in source
    assert 'st.session_state["studio_chat_nonce"] = chat_nonce + 1' in source
    assert 'st.session_state["studio_question"] = ""' not in source


def test_canva_redesign_keeps_functional_state_and_refresh_paths() -> None:
    source = _source()

    # Navigation and UI state must remain Streamlit-session based rather than
    # introducing a separate frontend state machine that could drift from the service.
    assert "st.session_state[\"studio_nav\"]" in source
    assert "st.session_state[\"console_answer\"]" in source
    assert "get_system.clear()" in source
    assert "Live refresh" in source
    assert "time.sleep(1.5)" in source


def test_settings_and_ingestion_have_input_guards() -> None:
    source = _source()

    assert "Chunk overlap must be smaller than chunk size." in source
    assert "No PDF files were found" in source
    assert "Folder does not exist" in source
    assert "if not ready" in source
    assert "disabled=not ready" in source
    assert "path.suffix.lower() == \".pdf\"" in source
