import streamlit as st

from rag_project.app import canva_exact_ui
from rag_project.app.canva_exact_ui import main as exact_main
from rag_project.security import (
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
/* One normal document flow: no floating canvas and no second page scroll layer. */
:root{--bookrag-sidebar-width:252px;--bookrag-content-max:1440px}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{min-height:100%;overflow-x:hidden!important}
[data-testid="stAppViewContainer"]{overflow:visible!important}
[data-testid="stAppViewContainer"]>section.main{min-width:0!important}
.studio-shell{display:none!important}
.block-container{position:relative!important;box-sizing:border-box!important;width:calc(100% - var(--bookrag-sidebar-width))!important;max-width:var(--bookrag-content-max)!important;margin-left:var(--bookrag-sidebar-width)!important;margin-right:0!important;padding:32px 32px 64px!important;min-width:0!important}
[data-testid="stSidebar"]{position:fixed!important;left:0!important;top:0!important;right:auto!important;bottom:0!important;width:var(--bookrag-sidebar-width)!important;min-width:var(--bookrag-sidebar-width)!important;height:100vh!important;max-height:100vh!important;overflow:hidden!important;z-index:100!important}
[data-testid="stSidebar"]>div:first-child{box-sizing:border-box!important;height:100%!important;max-height:100%!important;overflow-y:auto!important;overflow-x:hidden!important}
[data-testid="stHorizontalBlock"],[data-testid="stVerticalBlock"],[data-testid="stColumn"]{min-width:0!important;max-width:100%!important;box-sizing:border-box!important}
/* Regression guards for the original fixed Canva layout. */
.index-grid,.row-grid{display:grid!important;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)!important;gap:16px!important;width:100%!important}
.index-panel,.settings-panel,.query-panel,.health-panel,.inspector-panel{width:100%!important;max-width:100%!important;height:auto!important}
.data-wrap{max-width:100%!important;overflow:auto!important}.answer,.glass-note{max-width:100%!important;overflow-wrap:anywhere!important;word-break:break-word!important}
@media(max-width:1200px){:root{--bookrag-sidebar-width:220px}.block-container{padding:28px 22px 56px!important}.index-grid,.row-grid{grid-template-columns:minmax(0,1fr)!important}}
@media(max-width:760px){.block-container{width:100%!important;max-width:100%!important;margin-left:0!important;padding:20px 14px 44px!important}.index-grid,.row-grid{grid-template-columns:1fr!important}}
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
        clean = dict(updates or {})
        if "ollama_base_url" in clean:
            clean["ollama_base_url"] = validate_ollama_url(clean["ollama_base_url"])
        for key in ("incoming_dir", "processed_dir", "failed_dir", "archive_dir", "vector_db_dir", "log_dir", "ingestion_db_path"):
            if key in clean:
                clean[key] = validate_storage_path(system.settings.project_root, clean[key], key)
        return original_apply(clean)

    def guarded_answer(question, metadata_filter=None):
        return original_answer(validate_query(question), metadata_filter)

    system.clear_pdf_data = guarded_clear
    system.apply_settings_in_place = guarded_apply
    system.answer = guarded_answer
    system._bookrag_security_wrapped = True
    return system


# Preserve the Streamlit cache-resource control surface used by the UI's
# "Recreate runtime" action after installing the security facade.
_secure_system.clear = getattr(_ORIGINAL_GET_SYSTEM, "clear", lambda: None)
canva_exact_ui.get_system = _secure_system


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


canva_exact_ui.save_pdf = _secure_save_pdf
canva_exact_ui.start_ingestion = _secure_start_ingestion
canva_exact_ui.ollama_health = _secure_ollama_health


def main() -> None:
    st.set_page_config(page_title="BookRAG Studio", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
    if not require_auth():
        return
    original_page_config = canva_exact_ui.st.set_page_config
    canva_exact_ui.st.set_page_config = lambda *args, **kwargs: None
    try:
        exact_main()
    finally:
        canva_exact_ui.st.set_page_config = original_page_config
    st.markdown(_RESPONSIVE_RUNTIME_CSS, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
