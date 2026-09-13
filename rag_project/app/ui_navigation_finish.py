from __future__ import annotations

import streamlit as st

from rag_project.app import bookrag_ui as ui


_NAV_GROUPS = (
    ("Workspace", ("Home", "Documents", "Live Processing", "Ask BookRAG")),
    ("Research tools", ("Inspector", "System", "Settings")),
)

_INTERACTION_CSS = r'''<style>
[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background:#122235!important;
  border-color:#2d4a63!important;
  color:#f4f8fb!important;
  box-shadow:inset 3px 0 0 #70e1d6,0 6px 16px rgba(0,0,0,.12)!important;
}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{
  background:#162a40!important;
  border-color:#3b5e7b!important;
  transform:none!important;
}
.chip-processing{border-color:rgba(94,230,161,.30)!important;color:#9decc0!important;background:rgba(94,230,161,.06)!important}
.chip-ready{border-color:rgba(130,152,255,.30)!important;color:#aebcff!important;background:rgba(130,152,255,.06)!important}
.chip-idle{border-color:#26364b!important;color:#7f90a4!important;background:#0b121c!important}
.dot-processing{background:#5ee6a1!important;box-shadow:0 0 11px rgba(94,230,161,.8)!important}
.dot-ready{background:#8298ff!important;box-shadow:0 0 10px rgba(130,152,255,.65)!important}
.dot-idle{background:#62758c!important;box-shadow:none!important}
</style>'''


def _state(system) -> tuple[str, str, str]:
    active = len(ui.active_docs(system))
    ready = len(ui.ready_docs(system))
    if active or ui._job_running():
        return "PROCESSING", f"{active} processing", "processing"
    if ready:
        return "READY", f"{ready} ready", "ready"
    return "IDLE", "Workspace idle", "idle"


def _sidebar(system) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brandmark">BR</div>'
            '<div><div class="brandname">BookRAG Medical</div>'
            '<div class="brandsub">Evidence-first research workspace</div></div></div>',
            unsafe_allow_html=True,
        )
        current = st.session_state.get("bookrag_page", "Home")
        for group, items in _NAV_GROUPS:
            st.markdown(f'<div class="navgroup">{group}</div>', unsafe_allow_html=True)
            for item in items:
                active = item == current
                marker = "●" if active else "○"
                if st.button(
                    f"{marker}  {item}",
                    key=f"premium_nav_{item}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    ui._navigate(item)
        st.markdown(
            '<div class="sidefoot"><b>Local & private</b><br>'
            'Documents, retrieval and generation remain inside the configured local runtime.</div>',
            unsafe_allow_html=True,
        )


def _topbar(system, title: str) -> None:
    state, label, state_class = _state(system)
    markup = (
        '<div class="top">'
        '<div><div class="crumb">BookRAG · Research workspace</div>'
        f'<div class="title">{ui._esc(title)}</div></div>'
        f'<div class="topright"><div class="chip chip-{state_class}" title="{state}">'
        f'<span class="dot dot-{state_class}"></span>{ui._esc(label)}</div></div></div>'
    )
    st.markdown(markup, unsafe_allow_html=True)
    ui.command_palette()


def apply() -> None:
    """Install the final navigation/header interaction contract."""
    st.markdown(_INTERACTION_CSS, unsafe_allow_html=True)
    ui.sidebar = _sidebar
    ui.topbar = _topbar


__all__ = ["apply"]
