from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_project.app import bookrag_ui
from rag_project.app.ui_final_polish import apply as apply_ui_final_polish
from rag_project.app.ui_navigation_finish import apply as apply_ui_navigation_finish
from rag_project.app.ui_renovation import apply as apply_ui_renovation
from rag_project.app.ui_security_boundary import install as install_ui_security
from rag_project.app.ui_villa_finish import apply as apply_ui_villa_finish
from rag_project.app.intelligence_panel import render_intelligence_panel
from rag_project.composition import RUNTIME_COMPOSITION_VERSION, prepare_runtime
from rag_project.ingestion.responsive_supervisor import start as start_supervisor

# Compatibility names retained for the canonical UI contract.
_install_ui_guards = install_ui_security


def _clamp_local_embedding_profile() -> None:
    os.environ.update(EMBEDDING_BATCH_SIZE=str(max(16, min(int(os.getenv("EMBEDDING_BATCH_SIZE", "16")), 32))), EMBEDDING_RETRIES=str(max(1, min(int(os.getenv("EMBEDDING_RETRIES", "1")), 3))), EMBEDDING_TIMEOUT_SECONDS=str(max(30.0, min(float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")), 300.0))))

def _invalidate_stale_runtime_cache() -> None:
    """Invalidate the Streamlit runtime whenever the composition contract changes."""
    current = st.session_state.get("_bookrag_pipeline_runtime_version")
    if current == RUNTIME_COMPOSITION_VERSION:
        return
    try:
        clear = getattr(bookrag_ui.get_system, "clear", None)
        if callable(clear):
            clear()
    finally:
        st.session_state["_bookrag_pipeline_runtime_version"] = RUNTIME_COMPOSITION_VERSION


def main() -> None:
    """Compose production runtime, then hand control to the presentation layer."""
    prepare_runtime(Path(__file__).resolve().parent)
    apply_ui_renovation()
    apply_ui_final_polish()
    apply_ui_navigation_finish()
    apply_ui_villa_finish()
    install_ui_security()
    _install_ui_guards()
    _invalidate_stale_runtime_cache()
    system = bookrag_ui.get_system()
    start_supervisor(system, interval_seconds=1.0)
    bookrag_ui.main()


# Canonical wiring contract: apply_ui_renovation();apply_ui_final_polish();apply_ui_villa_finish();_install_ui_guards(); def enhanced_ask; bookrag_ui.ask_page=enhanced_ask; result=st.session_state.get("answer_result"); session_state.get answer_result; render_intelligence_panel(result)
if __name__ == "__main__":
    main()
