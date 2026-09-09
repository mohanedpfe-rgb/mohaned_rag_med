from __future__ import annotations

from pathlib import Path


def test_studio_ui_uses_composition_root_and_real_actions() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rag_project" / "app" / "studio_ui.py").read_text(encoding="utf-8")

    assert "create_rag_system" in source
    assert "system.answer" in source
    assert "system.ingest_directory" in source
    assert "system.clear_pdf_data" in source
    assert "system.verify_index" in source
    assert "system.vector_store.compatibility_report" in source
    assert "st.rerun()" in source


def test_upload_deduplication_is_hash_based() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rag_project" / "app" / "studio_ui.py").read_text(encoding="utf-8")

    assert "hashlib.sha256" in source
    assert "saved_pdf_hashes" in source
    assert "if digest in seen" in source


def test_canva_studio_information_architecture_is_present() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rag_project" / "app" / "studio_ui.py").read_text(encoding="utf-8")

    # The Canva concept is a single Studio shell with these primary destinations.
    for label in ("Overview", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"):
        assert f'"{label}"' in source
    assert "BookRAG Studio" in source
    assert "Start all chunks" in source
    assert "system.apply_settings_in_place" in source
    assert "console_answer" in source
    assert "Page checkpoints" in source
    assert "Recent process events" in source


def test_canva_redesign_keeps_functional_state_and_refresh_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rag_project" / "app" / "studio_ui.py").read_text(encoding="utf-8")

    # Navigation and UI state must remain Streamlit-session based rather than
    # introducing a separate frontend state machine that could drift from the service.
    assert "st.session_state[\"studio_nav\"]" in source
    assert "st.session_state[\"console_answer\"]" in source
    assert "get_system.clear()" in source
    assert "Live refresh" in source
