from __future__ import annotations

from pathlib import Path


def test_compatibility_ui_entrypoint_targets_canonical_production_app() -> None:
    source = (Path(__file__).resolve().parents[1] / "rag_project" / "app" / "ui.py").read_text(encoding="utf-8")
    assert "from app import main" in source
    assert "from rag_project.app.dev_ui import main" not in source


def test_production_app_wires_intelligence_panel_after_answer_generation() -> None:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "from rag_project.app.intelligence_panel import render_intelligence_panel" in source
    assert "session_state.get" in source and "answer_result" in source
    assert "render_intelligence_panel(result)" in source
    assert "bookrag_ui.ask_page=enhanced_ask" in source
