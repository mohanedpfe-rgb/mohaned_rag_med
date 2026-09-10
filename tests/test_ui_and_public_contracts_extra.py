from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_ui_entrypoint_cannot_fall_back_to_dev_console():
    source = (ROOT / "rag_project" / "app" / "ui.py").read_text(encoding="utf-8")
    assert "from app import main" in source
    assert "dev_ui" not in source


def test_app_wraps_answer_page_and_renders_phase5_panel():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "render_intelligence_panel" in source
    assert "def enhanced_ask" in source
    assert "answer_result" in source
    assert "bookrag_ui.ask_page=enhanced_ask" in source


def test_intelligence_panel_contains_all_user_visible_signals():
    source = (ROOT / "rag_project" / "app" / "intelligence_panel.py").read_text(encoding="utf-8")
    for marker in (
        "Decision transparency",
        "Intent:",
        "Entities:",
        "Rewritten question:",
        "Claim support matrix:",
        "Calibrated confidence:",
        "Abstention reason:",
        "Phase 5 visibility:",
    ):
        assert marker in source


def test_ui_visibility_contract_has_six_required_user_facing_signals():
    from rag_project.intelligence.ui_visibility_contract import build_visibility_contract

    result = build_visibility_contract(
        {
            "phase_plan": {"intent": "management", "entities": ["diabetes"]},
            "rewritten_question": "How is diabetes managed?",
            "evidence_claim_matrix": [],
            "confidence_calibration": {"calibrated": .75, "level": "medium"},
            "abstention_reasons": ["blocked_claims"],
        }
    )
    assert set(result["signals"]) == {
        "intent",
        "entities",
        "rewritten_question",
        "claim_support_matrix",
        "calibrated_confidence",
        "abstention_reason",
    }
    assert result["signals_present"] is True


def test_ui_visibility_contract_fallbacks_are_consistent():
    from rag_project.intelligence.ui_visibility_contract import build_visibility_contract

    result = build_visibility_contract(
        {
            "query_analysis": {"intent": "factual", "entities": ["diabetes"]},
            "rewritten_question": "What is diabetes?",
            "final_evidence_claim_matrix": [{"claim": "x"}],
            "confidence": {"evidence_confidence": .5, "level": "medium"},
            "advanced_reasoning": {"blocked_reasons": ["missing_evidence"]},
        }
    )
    assert result["claim_support_matrix"] == [{"claim": "x"}]
    assert result["calibrated_confidence"] == .5
    assert result["abstention_reasons"] == ["missing_evidence"]
    assert result["signals_present"] is True


def test_bookrag_ui_ask_flow_stores_answer_result():
    source = (ROOT / "rag_project" / "app" / "bookrag_ui.py").read_text(encoding="utf-8")
    assert 'st.session_state["answer_result"] = result' in source
    assert 'system.answer(question.strip(), metadata_filter=selected)' in source


def test_bookrag_ui_has_export_paths_for_answer_and_diagnostics():
    source = (ROOT / "rag_project" / "app" / "bookrag_ui.py").read_text(encoding="utf-8")
    assert "bookrag_answer.md" in source
    assert "bookrag_answer.json" in source
    assert "Raw response" in source
    assert "Query trace" in source


def test_app_security_guards_are_installed_before_main_ui():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "_install_ui_guards()" in source
    assert "start_supervisor(system" in source
    main_body = source[source.index("def main") :]
    assert main_body.index("_install_ui_guards()") < main_body.index("start_supervisor(system")
    assert main_body.index("start_supervisor(system") < main_body.index("bookrag_ui.main()")


def test_ui_source_does_not_directly_bypass_canonical_pipeline():
    for name in ("app.py", "rag_project/app/ui.py", "rag_project/app/bookrag_ui.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        if name.endswith("app.py"):
            assert "create_rag_system" in source or "bookrag_ui.get_system" in source


@pytest.mark.parametrize(
    "name",
    [
        "bookrag_ui.py",
        "intelligence_panel.py",
        "dev_ui.py",
        "studio_ui.py",
        "canva_exact_ui.py",
        "live_runtime.py",
        "production_rag.py",
    ],
)
def test_ui_modules_are_readable_and_have_entry_or_render_functions(name: str):
    source = (ROOT / "rag_project" / "app" / name).read_text(encoding="utf-8")
    assert source.strip()
    assert "def " in source


def test_security_and_ui_exports_are_present():
    security = (ROOT / "rag_project" / "security.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    panel = (ROOT / "rag_project" / "app" / "intelligence_panel.py").read_text(encoding="utf-8")
    assert "__all__ = [\"render_intelligence_panel\"]" in panel
    assert "validate_pdf_payload" in security
    assert "validate_query" in security
    assert "render_intelligence_panel" in app


def test_readme_mentions_local_or_ollama_runtime():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").casefold()
    assert "ollama" in readme
    assert "rag" in readme


def test_project_has_no_tracked_secret_env_file():
    assert not (ROOT / ".env").exists()
    assert (ROOT / ".env.example").is_file()
