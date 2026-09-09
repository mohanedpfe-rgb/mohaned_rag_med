from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
UI = (ROOT / "rag_project" / "app" / "canva_exact_ui.py").read_text(encoding="utf-8")
SECURITY = (ROOT / "rag_project" / "security.py").read_text(encoding="utf-8")


def test_runtime_layout_uses_single_normal_scroll_flow():
    assert '.studio-shell{display:none!important}' in APP
    assert '[data-testid="stAppViewContainer"]{overflow:visible!important}' in APP
    assert 'width:100%!important' in APP
    assert 'position:fixed!important;left:0!important;top:0!important' in APP
    assert '.index-grid,.row-grid{' in APP
    assert 'grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)!important' in APP


def test_all_navigation_pages_have_real_dispatch_targets():
    assert 'elif page == "Documents": documents(system)' in UI
    assert 'elif page == "Index them": ingestion(system, index_mode=True)' in UI
    assert 'elif page == "Ingestion": ingestion(system)' in UI
    for page in ("Inspector", "Settings", "Chat", "Health", "Background"):
        assert f'page == "{page}"' in UI


def test_navigation_controls_are_real_actions_not_decorative_markup():
    for key in ("nav_Overview", "nav_Documents", "nav_Index them", "nav_Ingestion", "nav_Inspector", "nav_Settings"):
        assert f'key=f"{key}"' in UI
    assert 'key="top_chat"' in UI
    assert 'key="top_health"' in UI
    assert 'key="top_refresh"' in UI
    assert 'st.session_state["studio_nav"]' in UI


def test_shared_components_use_one_cached_system_and_one_job_registry():
    assert '@st.cache_resource(show_spinner=False)' in UI
    assert 'system = get_system(); css(); sidebar(system); topbar()' in UI
    assert 'registry = get_jobs()' in UI
    assert 'start_ingestion(system' in UI


def test_health_page_uses_production_readiness_report():
    assert 'report = system.health_report()' in UI
    assert '"Production contract"' in UI
    assert '"Index audit"' in UI


def test_login_path_continues_after_success_and_default_password_exists():
    assert 'if not require_auth():' in APP
    assert '        return' in APP
    assert 'DEFAULT_LOCAL_PASSWORD = "becheikh_mohaned_med"' in SECURITY
    assert 'return True' in SECURITY
    assert 'st.rerun()' not in SECURITY[SECURITY.index('def require_auth'):SECURITY.index('def clear_confirmation_ui')]


def test_recreate_runtime_keeps_cache_clear_callable_after_security_facade():
    assert '_secure_system.clear = getattr(_ORIGINAL_GET_SYSTEM, "clear", lambda: None)' in APP
    assert 'get_system.clear()' in UI


def test_cleanup_has_one_confirmation_source_and_secure_backend_guard():
    assert 'key="bookrag_clear_phrase"' in UI
    assert 'phrase.strip() != "CLEAR ALL PDF DATA"' in UI
    assert 'require_clear_confirmation()' in SECURITY
    assert 'clear_confirmation_ui()' not in APP


def test_ui_source_has_no_unbounded_fixed_panel_override_in_runtime_css():
    start = APP.index('_RESPONSIVE_RUNTIME_CSS')
    css = APP[start:]
    assert '.studio-shell{display:none!important}' in css
    assert 'grid-template-columns:802px' not in css
    assert 'width:1441px' not in css
    assert 'height:855px' not in css
