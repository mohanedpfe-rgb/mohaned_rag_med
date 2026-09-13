from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_project.app import bookrag_ui
from rag_project.app.ui_final_polish import apply as apply_ui_final_polish
from rag_project.app.ui_renovation import apply as apply_ui_renovation
from rag_project.app.ui_security_boundary import install as install_ui_security
from rag_project.app.ui_villa_finish import apply as apply_ui_villa_finish
from rag_project.composition import RUNTIME_COMPOSITION_VERSION, prepare_runtime
from rag_project.ingestion.responsive_supervisor import start as start_supervisor


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
    apply_ui_villa_finish()
    install_ui_security()
    _invalidate_stale_runtime_cache()
    system = bookrag_ui.get_system()
    start_supervisor(system, interval_seconds=1.0)
    bookrag_ui.main()


if __name__ == "__main__":
    main()
