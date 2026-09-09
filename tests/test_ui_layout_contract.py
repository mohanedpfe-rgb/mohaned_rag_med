from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
UI = (ROOT / "rag_project" / "app" / "canva_exact_ui.py").read_text(encoding="utf-8")


def test_runtime_layout_uses_single_normal_scroll_flow():
    assert '.studio-shell{display:none!important}' in APP
    assert '[data-testid="stAppViewContainer"]{overflow:visible!important}' in APP
    assert 'width:100%!important' in APP
    assert 'position:fixed!important;\n left:0!important;\n top:0!important' in APP
    assert '.index-grid,.row-grid{' in APP
    assert 'grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)!important' in APP


def test_all_navigation_pages_have_real_dispatch_targets():
    assert 'elif page == "Documents": documents(system)' in UI
    assert 'elif page == "Index them": ingestion(system, index_mode=True)' in UI
    assert 'elif page == "Ingestion": ingestion(system)' in UI
    for page in ("Inspector", "Settings", "Chat", "Health", "Background"):
        assert f'page == "{page}"' in UI


def test_health_page_uses_production_readiness_report():
    assert 'report = system.health_report()' in UI
    assert '"Production contract"' in UI
    assert '"Index audit"' in UI


def test_ui_source_has_no_unbounded_fixed_panel_override_in_runtime_css():
    start = APP.index('_RESPONSIVE_RUNTIME_CSS')
    css = APP[start:]
    assert '.studio-shell{display:none!important}' in css
    assert 'grid-template-columns:802px' not in css
    assert 'width:1441px' not in css
    assert 'height:855px' not in css
