from __future__ import annotations

import streamlit as st

from rag_project.app import bookrag_ui as ui


_NAV_GROUPS = (
    ("Workspace", ("Home", "Documents", "Live Processing", "Ask BookRAG")),
    ("Research tools", ("Inspector", "System", "Settings")),
)


def _state(system) -> tuple[str, str, str, str]:
    active = len(ui.active_docs(system))
    ready = len(ui.ready_docs(system))
    running = bool(active or ui._job_running())
    if running:
        return "PROCESSING", f"{active} processing", "processing", "processing"
    if ready:
        return "READY", f"{ready} ready", "ready", "ready"
    return "IDLE", "Workspace idle", "idle", "idle"


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
    state, label, state_class, _ = _state(system)
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
    ui.sidebar = _sidebar
    ui.topbar = _topbar
    ui.renovation_sidebar = _sidebar if hasattr(ui, "renovation_sidebar") else getattr(ui, "sidebar", None)
    ui.renovation_topbar = _topbar if hasattr(ui, "renovation_topbar") else getattr(ui, "topbar", None)


__all__ = ["apply"]
