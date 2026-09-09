import streamlit as st

from rag_project.app import canva_exact_ui
from rag_project.security import (
    clear_confirmation_ui,
    require_auth,
    require_clear_confirmation,
    validate_ollama_url,
    validate_pdf_payload,
    validate_query,
    validate_storage_path,
)

_EXACT_RUNTIME_CSS = """
<style>
/* Fixed 1920x1080 calibration from Canva reference. */
.studio-shell{position:fixed!important;left:91px!important;top:112px!important;width:1738px!important;height:855px!important;z-index:0!important}
.block-container{position:relative!important;width:1441px!important;max-width:1441px!important;margin-left:342px!important;padding:23px 46px 56px!important}
.topline{position:fixed!important;left:388px!important;top:135px!important;width:1395px!important;height:45px!important;z-index:10!important;margin:0!important}
.title-block{position:fixed!important;left:388px!important;top:217px!important;width:760px!important;z-index:8!important;margin:0!important}
.index-grid{position:fixed!important;left:388px!important;top:360px!important;width:1395px!important;height:202px!important;z-index:7!important;margin:0!important}
.index-panel{position:fixed!important;left:388px!important;top:360px!important;width:802px!important;height:202px!important;box-sizing:border-box!important}
.settings-panel{position:fixed!important;left:1210px!important;top:360px!important;width:573px!important;height:202px!important;box-sizing:border-box!important}
.index-metric{position:fixed!important;top:405px!important;height:158px!important;box-sizing:border-box!important;z-index:8!important}
[data-testid="stMarkdownContainer"] .index-metric:nth-of-type(1){left:731px!important;width:161px!important}
[data-testid="stMarkdownContainer"] .index-metric:nth-of-type(2){left:868px!important;width:162px!important}
[data-testid="stMarkdownContainer"] .index-metric:nth-of-type(3){left:1006px!important;width:161px!important}
.query-panel{position:fixed!important;left:388px!important;top:600px!important;width:802px!important;height:74px!important;box-sizing:border-box!important;z-index:7!important;overflow:visible!important}
.health-panel{position:fixed!important;left:388px!important;top:694px!important;width:802px!important;height:236px!important;box-sizing:border-box!important;z-index:7!important}
.inspector-panel{position:fixed!important;left:1210px!important;top:694px!important;width:573px!important;height:236px!important;box-sizing:border-box!important;z-index:8!important}
.query-panel .panel-title{margin-top:0!important}.health-panel .panel-head{margin-bottom:18px!important}.inspector-panel .panel-head{margin-bottom:18px!important}
[class*="st-key-exact_overview_q"]{position:fixed!important;left:408px!important;top:642px!important;width:760px!important;z-index:15!important}
[class*="st-key-exact_overview_ask"]{position:fixed!important;left:408px!important;top:748px!important;width:190px!important;z-index:16!important}
[data-testid="stSidebar"]{position:fixed!important;left:91px!important;top:112px!important;width:251px!important;height:855px!important;z-index:20!important}
[data-testid="stSidebar"] .brand{position:fixed!important;left:169px!important;top:154px!important;width:175px!important;height:45px!important;margin:0!important}
[data-testid="stSidebar"] .nav-label{display:none!important}
[class*="st-key-nav_Overview"],[class*="st-key-nav_Documents"],[class*="st-key-nav_Index them"],[class*="st-key-nav_Ingestion"],[class*="st-key-nav_Inspector"],[class*="st-key-nav_Settings"]{position:fixed!important;left:114px!important;width:252px!important;height:45px!important;z-index:25!important}
[class*="st-key-nav_Overview"]{top:202px!important}[class*="st-key-nav_Documents"]{top:270px!important}[class*="st-key-nav_Index them"]{top:315px!important}[class*="st-key-nav_Ingestion"]{top:382px!important}[class*="st-key-nav_Inspector"]{top:427px!important}[class*="st-key-nav_Settings"]{top:472px!important}
[class*="st-key-nav_secondary_Chat"],[class*="st-key-nav_secondary_Health"],[class*="st-key-nav_secondary_Background"]{position:fixed!important;left:114px!important;width:205px!important;height:38px!important;z-index:25!important}
[class*="st-key-nav_secondary_Chat"]{top:525px!important}[class*="st-key-nav_secondary_Health"]{top:565px!important}[class*="st-key-nav_secondary_Background"]{top:605px!important}
[class*="st-key-exact_uploads"]{position:fixed!important;left:114px!important;top:660px!important;width:205px!important;height:48px!important;z-index:26!important;overflow:hidden!important}
[class*="st-key-exact_incoming"]{position:fixed!important;left:114px!important;top:713px!important;width:205px!important;z-index:26!important}
[class*="st-key-exact_start"]{position:fixed!important;left:114px!important;top:762px!important;width:205px!important;z-index:26!important}
[class*="st-key-exact_recreate"]{position:fixed!important;left:114px!important;top:811px!important;width:205px!important;z-index:26!important}
[class*="st-key-bookrag_clear_phrase"]{position:fixed!important;left:114px!important;top:856px!important;width:205px!important;z-index:27!important}
[class*="st-key-exact_confirm"]{display:none!important}
[class*="st-key-exact_clear"]{position:fixed!important;left:114px!important;top:904px!important;width:205px!important;z-index:26!important}
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
        for key in ("incoming_dir", "processed_dir", "failed_dir", "archive_dir", "vector_db_dir", "log_dir", "ingestion_db_path"):
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
        canva_exact_ui.main()
    finally:
        canva_exact_ui.st.set_page_config = original_page_config
    st.markdown(_EXACT_RUNTIME_CSS, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
