from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_launcher_exposes_intelligence_panel() -> None:
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "from rag_project.app.intelligence_panel import render_intelligence_panel" in app_source
    assert "def enhanced_ask" in app_source
    assert 'st.session_state.get("answer_result")' in app_source
    assert "render_intelligence_panel(result)" in app_source
    assert 'bookrag_ui.ask_page=enhanced_ask' in app_source


def test_intelligence_panel_displays_required_runtime_signals() -> None:
    panel_source = (ROOT / "rag_project" / "app" / "intelligence_panel.py").read_text(encoding="utf-8")

    required_signals = (
        "pipeline_authority",
        "phase_plan",
        "rewritten_question",
        "evidence_claim_matrix",
        "confidence_calibration",
        "abstention_reasons",
        "two_stage_synthesis",
        "adaptive_retrieval",
        "medical_term_layer",
        "advanced_reasoning",
        "production_contract",
        "phase_5_intelligence_visibility",
    )
    for signal in required_signals:
        assert signal in panel_source, f"Missing UI intelligence signal: {signal}"


def test_intelligence_panel_is_expanded_by_default() -> None:
    panel_source = (ROOT / "rag_project" / "app" / "intelligence_panel.py").read_text(encoding="utf-8")
    assert 'st.expander("Intelligence pipeline · 5 phases", expanded=True)' in panel_source
