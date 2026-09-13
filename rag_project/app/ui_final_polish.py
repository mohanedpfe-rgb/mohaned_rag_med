from __future__ import annotations

from typing import Any

import streamlit as st

from rag_project.app import bookrag_ui as ui
from rag_project.app import ui_renovation as renovation


_PREMIUM_CSS = r'''<style>
/* -------------------------------------------------------------------------
   Premium finish system
   4px base rhythm · 42px desktop controls · 46px compact/mobile controls
   10/12/14/18px radii · restrained elevation · explicit semantic states
   ------------------------------------------------------------------------- */
:root{
  --bg:#070b12;--surface:#0e1622;--surface-raised:#121d2b;--surface-soft:#0a131f;
  --line:#223247;--line-soft:#1a293b;--line-strong:#334963;
  --text:#f4f7fb;--text-soft:#d9e2ec;--muted:#91a2b6;--faint:#687a91;
  --accent:#70e1d6;--accent-2:#8298ff;--good:#5ee6a1;--warn:#f2c96d;--bad:#ff788b;--info:#8bb4ff;
  --shadow-1:0 8px 22px rgba(0,0,0,.12);--shadow-2:0 18px 48px rgba(0,0,0,.20);
  --ease:cubic-bezier(.22,.8,.2,1);
}

/* Foundation and page geometry. */
html,body,[class*=css]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(900px 560px at 92% -5%,rgba(130,152,255,.12),transparent 64%),radial-gradient(720px 480px at 3% 0,rgba(112,225,214,.07),transparent 60%),var(--bg);color:var(--text)}
[data-testid=stHeader]{height:0;background:transparent}
.block-container{max-width:1480px;padding:30px 38px 88px}

/* Sidebar: quieter chrome, stronger grouping, consistent control geometry. */
[data-testid=stSidebar]{background:linear-gradient(180deg,#09111c 0%,#07101a 100%)!important;border-right:1px solid var(--line)!important}
[data-testid=stSidebar]>div:first-child{padding:20px 14px 28px!important}
[data-testid=stSidebar] .stButton{margin:2px 0!important}
[data-testid=stSidebar] .stButton>button{min-height:42px!important;border:1px solid transparent!important;background:transparent!important;color:#94a5b9!important;border-radius:11px!important;text-align:left!important;font-size:11px!important;font-weight:760!important;padding:0 13px!important;transition:background .16s var(--ease),border-color .16s var(--ease),color .16s var(--ease),transform .16s var(--ease)!important}
[data-testid=stSidebar] .stButton>button:hover{background:#111d2c!important;border-color:#253850!important;color:#f5f8fb!important;transform:translateX(1px)!important}
[data-testid=stSidebar] .stButton>button:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px!important}
.brand{padding:6px 8px 22px!important}
.brandmark{width:42px!important;height:42px!important;border-radius:13px!important;box-shadow:0 12px 30px rgba(112,225,214,.15)!important}
.brandname{font-size:15px!important;letter-spacing:-.02em!important}.brandsub{font-size:9px!important;line-height:1.45!important}
.navgroup{margin:18px 8px 7px!important;color:#61738a!important;letter-spacing:.16em!important}
.sidefoot{margin-top:20px!important;padding:15px 8px!important;border-top:1px solid var(--line-soft)!important}

/* Header and status rhythm. */
.top{min-height:40px!important;margin-bottom:14px!important;gap:16px!important}
.crumb{font-size:9px!important;letter-spacing:.16em!important;color:#667992!important}
.title{font-size:25px!important;line-height:1.15!important;letter-spacing:-.045em!important;margin-top:5px!important}
.topright{gap:8px!important}.chip{height:32px!important;padding:0 11px!important;border-radius:999px!important;background:rgba(14,22,34,.82)!important}

/* Hero proportions and optical balance. */
.hero{border-radius:20px!important;padding:31px!important;box-shadow:var(--shadow-2)!important}
.hero h1{font-size:37px!important;line-height:1.04!important;max-width:860px!important}
.hero p{font-size:12px!important;line-height:1.75!important;max-width:780px!important}
.hero-actions{margin-top:21px!important}
.hero-actions .stButton{margin-right:6px!important}

/* Shared section architecture. */
.section{padding:20px!important;margin-top:14px!important;border-radius:15px!important;box-shadow:var(--shadow-1)!important}
.sectionhead{margin-bottom:14px!important;gap:16px!important}
.sectiontitle{font-size:15px!important;line-height:1.25!important}
.sectionsub{font-size:11px!important;line-height:1.6!important;max-width:820px!important}

/* Metric cards: align numbers and labels on the same visual grid. */
.statgrid,.metricgrid{gap:11px!important}
.stat{min-height:108px!important;padding:16px!important;border-radius:13px!important}
.statlabel{font-size:9px!important;letter-spacing:.10em!important}.statvalue{font-size:29px!important;margin-top:7px!important}.stathint{font-size:10px!important;line-height:1.45!important}
.metric{padding:13px!important;border-radius:12px!important}.metric label{font-size:8px!important;letter-spacing:.09em!important}.metric b{font-size:16px!important}.metric small{font-size:9px!important;line-height:1.45!important}

/* Documents/evidence: more breathing room without increasing noise. */
.docrow,.evidence,.metric{border-color:var(--line-soft)!important;background:#0b1420!important;border-radius:12px!important}
.docrow{padding:14px!important;margin-top:9px!important}
.dochead,.evidencehead{gap:14px!important}
.docname{font-size:13px!important;line-height:1.35!important}.meta,.evidencemeta{font-size:10px!important;line-height:1.55!important}
.meter{height:5px!important;margin:12px 0 9px!important}
.microgrid{gap:8px!important}.micro{padding:9px!important;border-radius:9px!important}.micro span{font-size:8px!important;letter-spacing:.04em!important}.micro b{font-size:11px!important;line-height:1.35!important}

/* Workflow cards and page rows. */
.workflow{gap:10px!important}.stepcard{padding:14px!important;border-radius:11px!important}.stepnum{font-size:9px!important}.stepname{font-size:11px!important;line-height:1.35!important}.stepdesc{font-size:10px!important;line-height:1.55!important}
.page{gap:12px!important;padding:11px 0!important;font-size:10px!important;line-height:1.45!important}.page small{font-size:9px!important;line-height:1.4!important}

/* Semantic status system. */
.pill{display:inline-flex!important;align-items:center!important;justify-content:center!important;gap:6px!important;min-height:22px!important;border-radius:999px!important;border:1px solid currentColor!important;padding:4px 9px!important;font-size:9px!important;font-weight:850!important;line-height:1!important;white-space:nowrap!important}
.pill i{width:6px!important;height:6px!important;flex:0 0 6px!important;border-radius:50%!important;background:currentColor!important}
.pill-good{color:var(--good)!important;background:rgba(94,230,161,.08)!important}.pill-warn{color:var(--warn)!important;background:rgba(242,201,109,.08)!important}.pill-bad{color:var(--bad)!important;background:rgba(255,120,139,.08)!important}.pill-neutral{color:#9aabbe!important;background:rgba(154,171,190,.07)!important}

/* Buttons: 42px desktop / 46px touch, 10px radii, measured hover lift. */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:42px!important;padding:0 15px!important;border-radius:10px!important;border:1px solid #2b3d55!important;background:#111d2c!important;color:#e8eef5!important;font-size:10.5px!important;font-weight:790!important;letter-spacing:.005em!important;transition:background .16s var(--ease),border-color .16s var(--ease),box-shadow .16s var(--ease),transform .16s var(--ease),color .16s var(--ease)!important}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:#16253a!important;border-color:#4b627c!important;box-shadow:0 7px 18px rgba(0,0,0,.16)!important;transform:translateY(-1px)!important}
.stButton>button:active,.stDownloadButton>button:active,.stFormSubmitButton>button:active{transform:translateY(0)!important;box-shadow:none!important}
.stButton>button:disabled,.stDownloadButton>button:disabled,.stFormSubmitButton>button:disabled{opacity:.48!important;cursor:not-allowed!important;transform:none!important;box-shadow:none!important}
.stButton>button[kind=primary]{background:linear-gradient(135deg,#4d8796,#516ea8)!important;border-color:#6a9eb2!important;color:#fff!important;box-shadow:0 8px 22px rgba(81,110,168,.20)!important}
.stButton>button[kind=primary]:hover{background:linear-gradient(135deg,#5793a1,#5b78b4)!important;border-color:#85b5c0!important}
.stDownloadButton>button{background:#101a28!important;border-color:#30445d!important}

/* Inputs/selects/forms: one control language across the application. */
.stTextInput input,.stTextArea textarea,.stNumberInput input,[data-baseweb=select],.stDateInput input{background:#0a1420!important;color:#edf3fa!important;border:1px solid #293b53!important;border-radius:10px!important}
.stTextInput input,.stNumberInput input{min-height:42px!important}.stTextArea textarea{min-height:96px!important;padding-top:11px!important}
.stTextInput input::placeholder,.stTextArea textarea::placeholder{color:#64768c!important}
[data-baseweb=select]>div{min-height:42px!important;background:#0a1420!important;border-color:#293b53!important;border-radius:10px!important}
.stTextInput input:focus,.stTextArea textarea:focus,.stNumberInput input:focus{border-color:#5f9ea3!important;box-shadow:0 0 0 1px #5f9ea3!important}
.stTextInput input:focus-visible,.stTextArea textarea:focus-visible,.stNumberInput input:focus-visible,[data-baseweb=select]:focus-within,.stButton>button:focus-visible,.stDownloadButton>button:focus-visible,.stFormSubmitButton>button:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px!important}

/* Uploader, progress, expanders, alerts. */
.stFileUploader{padding:9px!important;border:1px dashed #3b5571!important;border-radius:12px!important;background:#0a1420!important}
.stFileUploader section{background:linear-gradient(180deg,#0d1725,#09131f)!important;border-radius:10px!important}
.stFileUploader section>div{min-height:88px!important}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent-2))!important}
.stExpander{border:1px solid #263950!important;border-radius:11px!important;background:#0b1421!important}
.stExpander details{background:#0b1421!important}
.stAlert{border:1px solid var(--line)!important;border-radius:10px!important;background:#101a28!important}

/* Answer/evidence typography: research content gets the largest reading measure. */
.answertext{font-size:14px!important;line-height:1.82!important;color:#e3eaf2!important;max-width:900px!important}
.snippet{font-size:11px!important;line-height:1.68!important;color:#9eadbd!important}

/* Live state semantics: no fake LIVE label when idle. */
.livebar{min-height:40px!important;padding:10px 12px!important;gap:8px!important;border-radius:11px!important;border-color:var(--line-strong)!important;background:rgba(12,20,32,.92)!important}
.livebar.state-processing .livepulse{background:var(--good)!important;box-shadow:0 0 0 5px rgba(94,230,161,.07),0 0 12px rgba(94,230,161,.55)!important}
.livebar.state-ready .livepulse{background:var(--accent-2)!important;box-shadow:0 0 0 5px rgba(130,152,255,.08),0 0 12px rgba(130,152,255,.55)!important}
.livebar.state-idle .livepulse{background:#65758b!important;box-shadow:none!important}
.livebar.state-ready{background:rgba(17,28,45,.88)!important}.livebar.state-idle{background:rgba(11,18,29,.9)!important;color:#7f90a5!important}

/* Empty and diagnostic surfaces. */
.empty{padding:36px!important;border-radius:12px!important;line-height:1.55!important}
.timeline{padding-left:16px!important}.event{font-size:10px!important;line-height:1.5!important;padding-bottom:15px!important}.event small{font-size:9px!important}

/* Motion and accessibility. */
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible,.stFormSubmitButton>button:focus-visible{outline-offset:3px!important}
@media(prefers-reduced-motion:reduce){
  *,*::before,*::after{scroll-behavior:auto!important;transition:none!important;animation:none!important}
  .stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{transform:none!important}
}

/* Responsive architecture. */
@media(max-width:1150px){
  .block-container{padding-left:26px!important;padding-right:26px!important}
  .statgrid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
  .twocol{grid-template-columns:1fr!important}
  .metricgrid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
  .hero h1{font-size:33px!important}
}
@media(max-width:760px){
  .block-container{padding:18px 12px 54px!important}
  .title{font-size:22px!important}.crumb{font-size:8px!important}
  .hero{padding:22px!important;border-radius:17px!important}.hero h1{font-size:27px!important;letter-spacing:-.035em!important}.hero p{font-size:11px!important;line-height:1.7!important}
  .hero-actions .stButton>button{min-height:46px!important}
  .section{padding:15px!important;border-radius:14px!important}
  .statgrid,.metricgrid,.workflow{grid-template-columns:1fr!important}
  .microgrid{grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:8px!important}
  .docrow{padding:12px!important}
  .page{grid-template-columns:1fr!important;gap:4px!important;padding:10px 0!important}
  .page>div{min-width:0!important;overflow-wrap:anywhere!important}
  .top{align-items:flex-start!important}.topright{display:none!important}
  .livebar{align-items:flex-start!important;flex-wrap:wrap!important}.livebar .grow{display:none!important}
  .stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:46px!important}
  .stTextInput input,.stNumberInput input,[data-baseweb=select]>div{min-height:46px!important}
}
</style>'''


def _coherent_css() -> None:
    base_css = getattr(_coherent_css, "_base_css", None)
    if callable(base_css):
        base_css()
    st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)


def _honest_live_indicator(system) -> None:
    """Distinguish actual processing from a ready or idle workspace."""
    active = len(ui.active_docs(system))
    ready = len(ui.ready_docs(system))
    running = bool(active or ui._job_running())
    if running:
        state, label, cls = "PROCESSING", f"{active} document(s) actively processing", "state-processing"
    elif ready:
        state, label, cls = "READY", f"{ready} document(s) ready for grounded research", "state-ready"
    else:
        state, label, cls = "IDLE", "Workspace idle", "state-idle"
    pulse = '<span class="livepulse"></span>'
    markup = f'<div class="livebar {cls}">{pulse}<b>{state}</b><span>{renovation.esc(label)}</span><span class="grow"></span>{renovation.status(state)}</div>'
    st.markdown(markup, unsafe_allow_html=True)


def _premium_command_palette() -> None:
    """Use an honest label: this is navigation search, not a fake keyboard shortcut."""
    with st.popover("Quick navigation"):
        st.caption("Jump to a BookRAG workspace")
        query = st.text_input("Find a page", placeholder="Home, Documents, Ask…", key="renovation_command")
        normalized = (query or "").strip().casefold()
        commands = {"home":"Home","documents":"Documents","processing":"Live Processing","ask":"Ask BookRAG","inspector":"Inspector","system":"System","settings":"Settings"}
        for key, page in commands.items():
            if normalized and normalized not in key and normalized not in page.casefold():
                continue
            if st.button(f"{page}  →", key=f"renovation_command_{key}", use_container_width=True):
                ui._navigate(page)


def apply() -> None:
    """Apply measured premium finish after renovation and before security wrappers capture UI callables."""
    _coherent_css._base_css = renovation.inject_css
    ui.css = _coherent_css
    renovation.live_indicator = _honest_live_indicator
    renovation.command_palette = _premium_command_palette
    ui.live_indicator = _honest_live_indicator
    ui.command_palette = _premium_command_palette


__all__ = ["apply"]
