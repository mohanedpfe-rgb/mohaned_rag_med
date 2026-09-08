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
