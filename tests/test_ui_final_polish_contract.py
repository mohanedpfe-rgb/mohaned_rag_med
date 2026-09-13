from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
POLISH = (ROOT / "rag_project/app/ui_final_polish.py").read_text(encoding="utf-8")
RENOVATION = (ROOT / "rag_project/app/ui_renovation.py").read_text(encoding="utf-8")


def test_final_polish_is_applied_before_security_wrappers():
    assert "from rag_project.app.ui_final_polish import apply as apply_ui_final_polish" in APP
    assert "apply_ui_renovation();apply_ui_final_polish();_install_ui_guards()" in APP


def test_semantic_status_colors_are_defined_after_theme_replacement():
    for marker in (".pill-good", ".pill-warn", ".pill-bad", ".pill-neutral"):
        assert marker in POLISH


def test_live_indicator_has_explicit_processing_ready_idle_states():
    for marker in ("state-processing", "state-ready", "state-idle", 'state = "PROCESSING"', 'state = "READY"', 'state = "IDLE"'):
        assert marker in POLISH


def test_accessibility_and_motion_rules_are_present():
    for marker in ("focus-visible", "prefers-reduced-motion", "min-height:46px", "disabled"):
        assert marker in POLISH


def test_renovated_application_still_has_responsive_layout_contract():
    for marker in ("@media(max-width:1150px)", "@media(max-width:760px)", "block-container", "use_container_width"):
        assert marker in RENOVATION or marker in POLISH
