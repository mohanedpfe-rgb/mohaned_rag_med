from __future__ import annotations

from typing import Any

import streamlit as st

from rag_project.app import bookrag_ui as ui
from rag_project.app import ui_renovation as renovation


_POLISH_CSS = r'''<style>
/* Final cohesion layer: semantic states, readability, touch targets, and responsive behavior. */
:root{
  --bg:#070b12;--surface:#0f1724;--surface-raised:#121d2d;--surface-soft:#0b1421;
  --line:#223148;--line-strong:#30445f;--text:#f4f7fb;--muted:#93a3b7;--faint:#6c7d93;
  --accent:#70e1d6;--accent-2:#8298ff;--good:#5ee6a1;--warn:#f2c96d;--bad:#ff788b;--info:#8bb4ff;
}

/* Semantic status system. The renovation replaces the old stylesheet, so these must be explicit here. */
.pill{display:inline-flex!important;align-items:center!important;gap:6px!important;border-radius:999px!important;border:1px solid currentColor!important;padding:4px 9px!important;font-size:9px!important;font-weight:850!important;line-height:1!important;white-space:nowrap!important}
.pill i{width:6px!important;height:6px!important;flex:0 0 6px!important;border-radius:50%!important;background:currentColor!important}
.pill-good{color:var(--good)!important;background:rgba(94,230,161,.08)!important}
.pill-warn{color:var(--warn)!important;background:rgba(242,201,109,.08)!important}
.pill-bad{color:var(--bad)!important;background:rgba(255,120,139,.08)!important}
.pill-neutral{color:#9aabbe!important;background:rgba(154,171,190,.07)!important}

/* Readability: the first renovation used compact diagnostic sizing; this pass restores comfortable body text. */
.sectionsub,.stepdesc,.snippet,.meta,.evidencemeta,.stathint,.sidefoot{font-size:11px!important;line-height:1.65!important}
.sectiontitle{font-size:15px!important}
.docname{font-size:13px!important}
.page{font-size:10.5px!important;line-height:1.5!important}
.page small,.metric small,.micro span{font-size:9px!important}
.micro b{font-size:11px!important}
.statlabel{font-size:10px!important}

/* Unified controls. */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:42px!important;transition:background .16s ease,border-color .16s ease,box-shadow .16s ease,transform .16s ease!important}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-1px)}
.stButton>button:disabled,.stDownloadButton>button:disabled{opacity:.48!important;cursor:not-allowed!important;transform:none!important}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible,.stFormSubmitButton>button:focus-visible,
.stTextInput input:focus-visible,.stTextArea textarea:focus-visible,.stNumberInput input:focus-visible,
[data-baseweb="select"]:focus-within{outline:2px solid var(--accent)!important;outline-offset:2px!important}
.stDownloadButton>button{background:#101b2a!important;border:1px solid #2b3d55!important;color:#e5edf5!important}

/* File upload surface: stronger drop-zone affordance without changing upload behavior. */
.stFileUploader{padding:9px!important}
.stFileUploader section{background:linear-gradient(180deg,#0d1725,#0a1320)!important;border-radius:10px!important}
.stFileUploader section>div{min-height:86px!important}

/* Alerts and expanders read as product surfaces, not framework defaults. */
.stAlert{border:1px solid var(--line)!important;background:#101a28!important}
.stExpander details{background:#0b1421!important}

/* Live indicator semantics: green is reserved for actual active work. */
.livebar{border-color:var(--line-strong)!important}
.livebar.state-ready .livepulse{background:var(--accent-2)!important;box-shadow:0 0 0 5px rgba(130,152,255,.08),0 0 12px rgba(130,152,255,.55)!important}
.livebar.state-idle .livepulse{background:#65758b!important;box-shadow:none!important}
.livebar.state-ready{background:rgba(17,28,45,.88)!important}
.livebar.state-idle{background:rgba(11,18,29,.9)!important;color:#7f90a5!important}

/* Mobile: use comfortable tap targets and prevent dense horizontal clipping. */
@media(max-width:760px){
  .block-container{padding-left:12px!important;padding-right:12px!important}
  .hero{padding:23px!important;border-radius:18px!important}
  .hero-actions .stButton>button{min-height:46px!important}
  .section{padding:15px!important;border-radius:14px!important}
  .top{align-items:flex-start!important}
  .microgrid{gap:8px!important}
  .docrow{padding:12px!important}
  .page{grid-template-columns:1fr!important;gap:4px!important;padding:10px 0!important}
  .page>div{min-width:0!important;overflow-wrap:anywhere!important}
  .livebar{align-items:flex-start!important;flex-wrap:wrap!important}
  .livebar .grow{display:none!important}
  .metricgrid{grid-template-columns:1fr!important}
  .stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:46px!important}
}

/* Respect reduced-motion preferences. */
@media(prefers-reduced-motion:reduce){
  *,*::before,*::after{scroll-behavior:auto!important;transition:none!important;animation:none!important}
  .stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{transform:none!important}
}
</style>'''


def _coherent_css() -> None:
    """Run the existing renovation theme, then apply the final semantic/accessibility layer."""
    base_css = getattr(_coherent_css, "_base_css", None)
    if callable(base_css):
        base_css()
    st.markdown(_POLISH_CSS, unsafe_allow_html=True)


def _honest_live_indicator(system) -> None:
    """Show LIVE only when work is actually active; READY and IDLE are distinct states."""
    active = len(ui.active_docs(system))
    ready = len(ui.ready_docs(system))
    running = active or ui._job_running()
    if running:
        state = "PROCESSING"
        label = f"{active} document(s) actively processing"
        cls = "state-processing"
    elif ready:
        state = "READY"
        label = f"{ready} document(s) ready for grounded research"
        cls = "state-ready"
    else:
        state = "IDLE"
        label = "Workspace idle"
        cls = "state-idle"
    pulse = '<span class="livepulse"></span>'
    html = f'<div class="livebar {cls}">{pulse}<b>{state}</b><span>{renovation.esc(label)}</span><span class="grow"></span>{renovation.status(state)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def apply() -> None:
    """Apply the final UI renovation layer after the main renovation, before security wrappers capture UI callables."""
    _coherent_css._base_css = renovation.inject_css
    ui.css = _coherent_css
    renovation.live_indicator = _honest_live_indicator


__all__ = ["apply"]
