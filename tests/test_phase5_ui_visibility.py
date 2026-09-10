from __future__ import annotations

from pathlib import Path


def test_phase5_visibility_contract_module_exposes_all_required_signals() -> None:
    source = (Path(__file__).resolve().parents[1] / "rag_project" / "intelligence" / "ui_visibility_contract.py").read_text(encoding="utf-8")
    for signal in (
        '"intent"',
        '"entities"',
        '"rewritten_question"',
        '"claim_support_matrix"',
        '"calibrated_confidence"',
        '"abstention_reasons"',
        '"signals_present"',
    ):
        assert signal in source


def test_intelligence_panel_renders_phase5_transparency_summary() -> None:
    source = (Path(__file__).resolve().parents[1] / "rag_project" / "app" / "intelligence_panel.py").read_text(encoding="utf-8")
    required = (
        "build_visibility_contract",
        "_render_phase5_transparency",
        "Decision transparency",
        "Calibrated confidence:",
        "Abstention reason:",
        "Rewritten question:",
        "Claim support matrix:",
        "Phase 5 visibility:",
    )
    for signal in required:
        assert signal in source, f"Missing Phase-5 UI implementation signal: {signal}"


def test_canonical_app_still_wires_the_panel_after_answer_generation() -> None:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "from rag_project.app.intelligence_panel import render_intelligence_panel" in source
    assert "result=st.session_state.get('answer_result')" in source or 'result=st.session_state.get("answer_result")' in source
    assert "render_intelligence_panel(result)" in source
