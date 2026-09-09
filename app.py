import streamlit as st

from rag_project.app import canva_exact_ui
from rag_project.app.canva_exact_ui import main as exact_main
from rag_project.security import (
    clear_confirmation_ui,
    register_session_upload,
    require_auth,
    require_clear_confirmation,
    validate_ollama_url,
    validate_pdf_payload,
    validate_query,
    validate_storage_path,
)

_RESPONSIVE_RUNTIME_CSS = """
<style>
/* The previous 1920x1080 fixed calibration caused panels to sit outside the
   browser viewport and let the page scroll underneath floating elements. */
:root{--bookrag-sidebar-width:252px;--bookrag-gap:24px;--bookrag-content-max:1440px}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{min-height:100%;overflow-x:hidden!important}
[data-testid="stAppViewContainer"]{overflow-y:auto!important}
[data-testid="stSidebar"]{position:relative!important;left:auto!important;top:auto!important;width:var(--bookrag-sidebar-width)!important;min-width:var(--bookrag-sidebar-width)!important;height:auto!important;max-height:none!important;z-index:5!important;overflow:visible!important}
[data-testid="stSidebar"]>div:first-child{height:auto!important;max-height:none!important;overflow-y:auto!important;overflow-x:hidden!important}
[data-testid="stSidebar"] .brand{position:relative!important;left:auto!important;top:auto!important;width:auto!important;height:auto!important;margin:32px 28px 24px!important}
[data-testid="stSidebar"] .nav-label{display:block!important}
.block-container{position:relative!important;width:min(calc(100vw - var(--bookrag-sidebar-width) - 2*var(--bookrag-gap)),var(--bookrag-content-max))!important;max-width:var(--bookrag-content-max)!important;margin:0 auto!important;padding:32px 28px 64px!important}

/* Cancel fixed-coordinate and fixed-width rules inherited from the exact UI. */
.studio-shell,.topline,.title-block,.index-grid,.index-panel,.settings-panel,.index-metric,.query-panel,.health-panel,.inspector-panel,.row-grid,
[class*="st-key-exact_overview_q"],[class*="st-key-exact_overview_ask"],
[class*="st-key-nav_Overview"],[class*="st-key-nav_Documents"],[class*="st-key-nav_Index them"],[class*="st-key-nav_Ingestion"],[class*="st-key-nav_Inspector"],[class*="st-key-nav_Settings"],
[class*="st-key-nav_secondary_Chat"],[class*="st-key-nav_secondary_Health"],[class*="st-key-nav_secondary_Background"],
[class*="st-key-exact_uploads"],[class*="st-key-exact_incoming"],[class*="st-key-exact_start"],[class*="st-key-exact_recreate"],[class*="st-key-bookrag_clear_phrase"],[class*="st-key-exact_confirm"],[class*="st-key-exact_clear"]{
 position:relative!important;left:auto!important;right:auto!important;top:auto!important;bottom:auto!important;
 width:auto!important;min-width:0!important;max-width:none!important;height:auto!important;max-height:none!important;
 margin:revert!important;z-index:auto!important;overflow:visible!important;box-sizing:border-box!important}
.studio-shell{width:100%!important}
.index-grid{display:grid!important;grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr)!important;gap:var(--bookrag-gap)!important}
.row-grid{display:grid!important;grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr)!important;gap:var(--bookrag-gap)!important}
.index-panel,.settings-panel,.query-panel,.health-panel,.inspector-panel{width:100%!important}
.index-metric{position:relative!important;top:auto!important}
[class*="st-key-nav_Overview"],[class*="st-key-nav_Documents"],[class*="st-key-nav_Index them"],[class*="st-key-nav_Ingestion"],[class*="st-key-nav_Inspector"],[class*="st-key-nav_Settings"]{width:100%!important;height:auto!important}
[class*="st-key-nav_secondary_Chat"],[class*="st-key-nav_secondary_Health"],[class*="st-key-nav_secondary_Background"]{width:100%!important;height:auto!important}
[class*="st-key-exact_uploads"],[class*="st-key-exact_incoming"],[class*="st-key-exact_start"],[class*="st-key-exact_recreate"],[class*="st-key-bookrag_clear_phrase"],[class*="st-key-exact_clear"]{width:100%!important}
[data-testid="stHorizontalBlock"],[data-testid="stVerticalBlock"],[data-testid="stColumn"]{min-width:0!important;max-width:100%!important}
@media (max-width:1100px){
 :root{--bookrag-sidebar-width:220px}
 .index-grid,.row-grid{grid-template-columns:1fr!important}
 .block-container{width:calc(100vw - var(--bookrag-sidebar-width))!important}
}
@media (max-width:760px){
 [data-testid="stSidebar"]{width:100%!important;min-width:0!important}
 .block-container{width:100%!important;max-width:100%!important;padding:20px 16px 48px!important}
}
</style>
"""

_ORIGINAL_GET_SYSTEM = canva_exact_ui.get_system
_ORIGINAL_SAVE_PDF = canva_exact_ui.save_pdf
_ORIGINAL_START_INGESTION = canva_exact_ui.start_ingestion
_ORIGINAL_OLLAMA_HEALTH = canva_exact_ui.ollama_health


def _secure_system():
    system = _ORIGINAL_GET_SYSTEM()
    if getattr(system, "_bookrag_security_wrapped", False):
        return system
    original_clear = system.clear_pdf_data
    original_apply = system.apply_settings_in_place
    original_answer = system.answer
    def guarded_clear() -> None:
        require_clear_confirmation()
        original_clear()
    def guarded_apply(updates):
        updates = dict(updates or {})
        if "ollama_base_url" in updates:
            updates["ollama_base_url"] = validate_ollama_url(updates["ollama_base_url"])
        for key in ("incoming_dir","processed_dir","failed_dir","archive_dir","vector_db_dir","log_dir","ingestion_db_path"):
            if key in updates:
                updates[key] = validate_storage_path(system.settings.project_root, updates[key], key)
        return original_apply(updates)
    def guarded_answer(question, metadata_filter=None):
        return original_answer(validate_query(question), metadata_filter)
    system.clear_pdf_data = guarded_clear
    system.apply_settings_in_place = guarded_apply
    system.answer = guarded_answer
    system._bookrag_security_wrapped = True
    return system


def _secure_save_pdf(incoming, name, content):
    system = _secure_system()
    safe_incoming = validate_storage_path(system.settings.project_root, incoming, "incoming folder")
    validate_pdf_payload(name, content)
    register_session_upload(len(content))
    return _ORIGINAL_SAVE_PDF(safe_incoming, name, content)


def _secure_start_ingestion(system, source_dir):
    safe_source = validate_storage_path(system.settings.project_root, source_dir, "incoming folder")
    return _ORIGINAL_START_INGESTION(system, str(safe_source))


def _secure_ollama_health(base_url):
    return _ORIGINAL_OLLAMA_HEALTH(validate_ollama_url(base_url))


canva_exact_ui.get_system = _secure_system
canva_exact_ui.save_pdf = _secure_save_pdf
canva_exact_ui.start_ingestion = _secure_start_ingestion
canva_exact_ui.ollama_health = _secure_ollama_health


def main() -> None:
    st.set_page_config(page_title="BookRAG Studio", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
    require_auth()
    clear_confirmation_ui()
    original_page_config = canva_exact_ui.st.set_page_config
    canva_exact_ui.st.set_page_config = lambda *args, **kwargs: None
    try:
        exact_main()
    finally:
        canva_exact_ui.st.set_page_config = original_page_config
    st.markdown(_RESPONSIVE_RUNTIME_CSS, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
