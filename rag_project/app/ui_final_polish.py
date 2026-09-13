from __future__ import annotations

import streamlit as st

from rag_project.app import bookrag_ui as ui
from rag_project.app import ui_renovation as renovation


_PREMIUM_CSS = r'''<style>
/* -------------------------------------------------------------------------
   BookRAG Medical — Premium Finish v2
   One authoritative presentation contract.
   4px base rhythm · 1480px content · 44px desktop controls · 48px touch
   10/12/14/18/24px radii · restrained elevation · explicit semantic states
   ------------------------------------------------------------------------- */
:root{
  --bg:#070b12;--surface:#0e1622;--surface-raised:#121d2b;--surface-soft:#0a131f;
  --line:#223247;--line-soft:#1a293b;--line-strong:#334963;
  --text:#f4f7fb;--text-soft:#d9e2ec;--muted:#91a2b6;--faint:#687a91;
  --accent:#70e1d6;--accent-2:#8298ff;--good:#5ee6a1;--warn:#f2c96d;--bad:#ff7885;--info:#8bb4ff;
  --shadow-1:0 8px 22px rgba(0,0,0,.11);--shadow-2:0 18px 48px rgba(0,0,0,.20);
  --shadow-3:0 28px 76px rgba(0,0,0,.30);--ease:cubic-bezier(.22,.8,.2,1);
}

/* ─────────────── Foundation ─────────────── */
html,body,[class*=css]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body{color:var(--text)}
.stApp{background:radial-gradient(1000px 620px at 92% -10%,rgba(130,152,255,.11),transparent 64%),radial-gradient(760px 520px at 4% 0%,rgba(112,225,214,.065),transparent 62%),linear-gradient(180deg,#070b12 0%,#080d15 100%);color:var(--text)}
[data-testid=stHeader]{height:0!important;background:transparent!important}
.block-container{max-width:1480px!important;padding:30px 38px 88px!important}
[data-testid=stVerticalBlock]>[data-testid=element-container]+[data-testid=element-container]{margin-top:2px!important}
[data-testid=stHorizontalBlock]{gap:12px!important}
[data-testid=stMarkdownContainer] p{color:var(--muted)}
.stCaption{color:var(--muted)!important;font-size:10px!important;line-height:1.55!important}

/* ─────────────── Sidebar ─────────────── */
[data-testid=stSidebar]{background:linear-gradient(180deg,#09111c 0%,#07101a 100%)!important;border-right:1px solid var(--line)!important}
[data-testid=stSidebar]>div:first-child{padding:21px 14px 28px!important}
[data-testid=stSidebar] .stButton{margin:2px 0!important}
[data-testid=stSidebar] .stButton>button{min-height:42px!important;padding:0 13px!important;border:1px solid transparent!important;border-radius:11px!important;background:transparent!important;color:#94a5b9!important;font-size:11px!important;font-weight:760!important;text-align:left!important;transition:background .16s var(--ease),border-color .16s var(--ease),color .16s var(--ease),transform .16s var(--ease)!important}
[data-testid=stSidebar] .stButton>button:hover{transform:translateX(1px)!important;background:#111d2c!important;border-color:#263a53!important;color:#f5f8fb!important}
[data-testid=stSidebar] .stButton>button:active{transform:none!important}
[data-testid=stSidebar] .stButton>button:focus-visible{outline:2px solid var(--accent)!important;outline-offset:3px!important}
.brand{display:flex!important;align-items:center!important;gap:11px!important;padding:5px 8px 23px!important}
.brandmark{width:42px!important;height:42px!important;flex:0 0 42px!important;border-radius:14px!important;background:linear-gradient(135deg,var(--accent),var(--accent-2))!important;box-shadow:0 12px 32px rgba(112,225,214,.16)!important;color:#061017!important;font-size:12px!important;font-weight:950!important;letter-spacing:-.04em!important}
.brandname{font-size:15px!important;font-weight:900!important;line-height:1.15!important;letter-spacing:-.025em!important}.brandsub{margin-top:4px!important;color:var(--faint)!important;font-size:9px!important;line-height:1.45!important}
.navgroup{margin:18px 8px 7px!important;color:#64768d!important;font-size:9px!important;font-weight:900!important;letter-spacing:.16em!important;text-transform:uppercase!important}
.sidefoot{margin-top:20px!important;padding:15px 8px!important;border-top:1px solid var(--line-soft)!important;color:#71839a!important;font-size:9px!important;line-height:1.65!important}

/* ─────────────── Page header / command surface ─────────────── */
.top{min-height:42px!important;margin-bottom:15px!important;gap:18px!important}.crumb{color:#657990!important;font-size:9px!important;font-weight:900!important;letter-spacing:.17em!important;line-height:1!important;text-transform:uppercase!important}.title{margin-top:6px!important;color:var(--text)!important;font-size:25px!important;font-weight:900!important;letter-spacing:-.045em!important;line-height:1.1!important}.topright{gap:8px!important}
.chip{min-height:32px!important;padding:0 11px!important;border:1px solid var(--line)!important;border-radius:999px!important;background:rgba(14,22,35,.78)!important;color:#9aaec2!important;font-size:9px!important;font-weight:760!important;display:inline-flex!important;align-items:center!important}.dot{width:6px!important;height:6px!important;margin-right:7px!important;border-radius:50%!important}

/* ─────────────── Hero ─────────────── */
.hero{position:relative!important;overflow:hidden!important;padding:32px!important;border:1px solid #2b3e57!important;border-radius:24px!important;background:radial-gradient(500px 240px at 100% 0%,rgba(112,225,214,.09),transparent 70%),linear-gradient(128deg,#0d1624 0%,#101d2e 58%,#142a35 100%)!important;box-shadow:var(--shadow-3)!important}.hero:after{content:"";position:absolute;right:-150px;top:-180px;width:420px;height:420px;border-radius:50%;background:radial-gradient(circle,rgba(130,152,255,.11),transparent 67%)}.hero .kicker,.hero h1,.hero p,.hero-actions{position:relative!important;z-index:1!important}.hero .kicker{color:var(--accent)!important;font-size:9px!important;font-weight:900!important;letter-spacing:.18em!important;text-transform:uppercase!important}.hero h1{max-width:900px!important;margin:10px 0 9px!important;color:#f7fafc!important;font-size:38px!important;font-weight:910!important;letter-spacing:-.055em!important;line-height:1.02!important}.hero p{max-width:820px!important;margin:0!important;color:#9eafc1!important;font-size:12px!important;line-height:1.75!important}.hero-actions{margin-top:22px!important}.hero-actions .stButton{margin-right:4px!important}

/* ─────────────── Shared sections ─────────────── */
.section{margin-top:14px!important;padding:20px!important;border:1px solid var(--line)!important;border-radius:16px!important;background:rgba(13,22,34,.91)!important;box-shadow:var(--shadow-1)!important}.sectionhead{margin-bottom:14px!important;gap:16px!important}.sectiontitle{color:var(--text)!important;font-size:15px!important;font-weight:900!important;letter-spacing:-.02em!important;line-height:1.25!important}.sectionsub{max-width:860px!important;margin-top:4px!important;color:#718399!important;font-size:10px!important;line-height:1.62!important}
.statgrid,.grid4,.metricgrid{gap:11px!important}.statgrid,.grid4{margin-top:13px!important}.stat{min-height:112px!important;padding:16px!important;border:1px solid var(--line)!important;border-radius:14px!important;background:linear-gradient(180deg,rgba(17,29,45,.96),rgba(10,18,29,.96))!important;box-shadow:0 7px 20px rgba(0,0,0,.07)!important}.statlabel,.metric label{color:#70839a!important;font-size:8px!important;font-weight:850!important;letter-spacing:.10em!important;line-height:1.3!important;text-transform:uppercase!important}.statvalue{margin-top:7px!important;color:#f2f6fa!important;font-size:29px!important;font-weight:920!important;letter-spacing:-.06em!important;line-height:1!important}.stathint{margin-top:5px!important;color:#5c6e84!important;font-size:9px!important;line-height:1.45!important}.metricgrid{margin-top:11px!important}.metric{min-width:0!important;padding:13px!important;border:1px solid var(--line-soft)!important;border-radius:12px!important;background:#0b1421!important}.metric b{display:block!important;margin-top:5px!important;overflow-wrap:anywhere!important;color:var(--text-soft)!important;font-size:16px!important;font-weight:860!important;line-height:1.25!important}.metric small{display:block!important;margin-top:4px!important;color:#5d6f86!important;font-size:9px!important;line-height:1.45!important}

/* ─────────────── Documents / evidence ─────────────── */
.docrow,.evidence{border:1px solid var(--line-soft)!important;background:#0b1421!important;border-radius:12px!important;transition:border-color .16s var(--ease),background .16s var(--ease),box-shadow .16s var(--ease)!important}.docrow{padding:14px!important;margin-top:9px!important}.docrow:hover,.evidence:hover{border-color:#29415b!important;background:#0c1624!important;box-shadow:0 7px 20px rgba(0,0,0,.08)!important}.dochead,.evidencehead{gap:14px!important}.docname,.evidencetitle{color:#e6edf4!important;font-size:12px!important;font-weight:850!important;line-height:1.35!important;overflow-wrap:anywhere!important}.meta,.evidencemeta{margin-top:4px!important;color:#6e8299!important;font-size:9px!important;line-height:1.55!important}.meter,.bar{overflow:hidden!important;height:5px!important;border-radius:999px!important;background:#19283a!important}.meter{margin:12px 0 9px!important}.meter i,.bar i{display:block!important;height:100%!important;border-radius:inherit!important;background:linear-gradient(90deg,var(--accent),var(--accent-2))!important}.microgrid{gap:8px!important}.micro{min-width:0!important;padding:9px!important;border:1px solid #1c2c40!important;border-radius:9px!important;background:#0d1725!important}.micro span{color:#60748b!important;font-size:8px!important;letter-spacing:.05em!important;text-transform:uppercase!important}.micro b{margin-top:3px!important;overflow-wrap:anywhere!important;color:#dbe5ee!important;font-size:10px!important;line-height:1.35!important}

/* ─────────────── Status language ─────────────── */
.pill{display:inline-flex!important;align-items:center!important;justify-content:center!important;gap:6px!important;min-height:22px!important;padding:4px 9px!important;border:1px solid currentColor!important;border-radius:999px!important;font-size:9px!important;font-weight:900!important;line-height:1!important;white-space:nowrap!important}.pill i{width:6px!important;height:6px!important;flex:0 0 6px!important;border-radius:50%!important;background:currentColor!important}.pill-good{color:var(--good)!important;background:rgba(94,230,161,.07)!important}.pill-warn{color:var(--warn)!important;background:rgba(242,201,109,.07)!important}.pill-bad{color:var(--bad)!important;background:rgba(255,120,139,.07)!important}.pill-neutral{color:#9aabbe!important;background:rgba(154,171,190,.06)!important}

/* ─────────────── Workflow / pipeline ─────────────── */
.workflow{gap:10px!important}.stepcard{min-height:120px!important;padding:14px!important;border:1px solid var(--line-soft)!important;border-radius:12px!important;background:#0b1421!important;transition:border-color .16s var(--ease),transform .16s var(--ease),box-shadow .16s var(--ease)!important}.stepcard:hover{transform:translateY(-1px)!important;border-color:#2b435e!important;box-shadow:var(--shadow-1)!important}.stepnum{color:var(--accent)!important;font-size:9px!important;font-weight:900!important;letter-spacing:.08em!important}.stepname{margin-top:5px!important;color:#e8eef4!important;font-size:11px!important;font-weight:860!important;line-height:1.35!important}.stepdesc{margin-top:5px!important;color:#75879c!important;font-size:9px!important;line-height:1.62!important}.pipeline{gap:5px!important;margin:13px 0!important}.pipeline .step{height:6px!important;border-radius:999px!important;background:#1a2a3d!important}.pipeline .step.on{background:linear-gradient(90deg,var(--accent),var(--accent-2))!important}

/* ─────────────── Controls ─────────────── */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:44px!important;padding:0 15px!important;border:1px solid #2c3f57!important;border-radius:10px!important;background:#111d2c!important;color:#e7eef5!important;font-size:10.5px!important;font-weight:800!important;letter-spacing:.005em!important;box-shadow:none!important;transition:background .16s var(--ease),border-color .16s var(--ease),box-shadow .16s var(--ease),transform .16s var(--ease),color .16s var(--ease)!important}.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{transform:translateY(-1px)!important;border-color:#536b86!important;background:#17263a!important;color:#f7fafc!important;box-shadow:0 8px 20px rgba(0,0,0,.14)!important}.stButton>button:active,.stDownloadButton>button:active,.stFormSubmitButton>button:active{transform:none!important;box-shadow:none!important}.stButton>button:disabled,.stDownloadButton>button:disabled,.stFormSubmitButton>button:disabled{opacity:.45!important;cursor:not-allowed!important;transform:none!important;box-shadow:none!important}.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{border-color:#6d9fb0!important;background:linear-gradient(135deg,#4e8897,#536eab)!important;color:#fff!important;box-shadow:0 8px 22px rgba(81,110,168,.19)!important}.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover{border-color:#8abac4!important;background:linear-gradient(135deg,#5896a4,#5d79b5)!important}.stDownloadButton>button{background:#101a28!important;border-color:#30445d!important}

/* ─────────────── Forms / selects / menus ─────────────── */
.stTextInput input,.stTextArea textarea,.stNumberInput input,.stDateInput input,[data-baseweb="select"]>div{min-height:44px!important;border:1px solid #293c54!important;border-radius:10px!important;background:#0a1420!important;color:#edf4f9!important;box-shadow:none!important}.stTextArea textarea{min-height:104px!important;padding-top:12px!important;line-height:1.55!important}.stTextInput input::placeholder,.stTextArea textarea::placeholder{color:#667a91!important}.stTextInput input:focus,.stTextArea textarea:focus,.stNumberInput input:focus,.stDateInput input:focus,[data-baseweb="select"]>div:focus-within{border-color:#5d9ba2!important;box-shadow:0 0 0 1px #5d9ba2!important}.stTextInput input:focus-visible,.stTextArea textarea:focus-visible,.stNumberInput input:focus-visible,.stDateInput input:focus-visible,[data-baseweb="select"]:focus-within,.stButton>button:focus-visible,.stDownloadButton>button:focus-visible,.stFormSubmitButton>button:focus-visible{outline:2px solid var(--accent)!important;outline-offset:3px!important}[data-baseweb="tag"]{min-height:25px!important;border:1px solid #304865!important;border-radius:8px!important;background:#16263b!important;color:#dfe8f0!important}[data-baseweb="tag"] span{font-size:9px!important}[role="listbox"]{padding:5px!important;border:1px solid var(--line-strong)!important;border-radius:12px!important;background:#0b1522!important;box-shadow:var(--shadow-3)!important}[role="option"]{min-height:36px!important;padding:8px 10px!important;border-radius:8px!important;color:#d9e3ed!important}[role="option"]:hover,[role="option"][aria-selected="true"]{background:#142237!important;color:#fff!important}[data-testid="stForm"]{padding:13px!important;border:1px solid var(--line-soft)!important;border-radius:14px!important;background:rgba(10,20,32,.74)!important;box-shadow:var(--shadow-1)!important}
[data-testid="stCheckbox"] label,[data-testid="stRadio"] label{color:#dbe5ee!important;font-size:10px!important;line-height:1.45!important}[data-testid="stCheckbox"] [data-baseweb="checkbox"]{margin-right:7px!important}[data-testid="stRadio"] [data-baseweb="radio"]{gap:7px!important}
[data-testid="stSlider"] [role="slider"]{box-shadow:0 0 0 3px rgba(112,225,214,.09)!important}[data-testid="stSlider"] [role="slider"]:focus-visible{outline:2px solid var(--accent)!important;outline-offset:3px!important}[data-testid="stSlider"] [data-baseweb="slider"]>div:first-child{height:4px!important}

/* ─────────────── Upload / progress / disclosure ─────────────── */
.stFileUploader{padding:9px!important;border:1px dashed #3b5571!important;border-radius:12px!important;background:#0a1420!important}.stFileUploader section{background:linear-gradient(180deg,#0d1725,#09131f)!important;border-radius:10px!important}.stFileUploader section>div{min-height:88px!important}[data-testid="stFileUploaderDropzone"]{min-height:92px!important;border:1px dashed #3b5571!important;border-radius:10px!important;background:linear-gradient(180deg,#0c1725,#09131f)!important;transition:border-color .16s var(--ease),background .16s var(--ease),box-shadow .16s var(--ease)!important}[data-testid="stFileUploaderDropzone"]:hover{border-color:#6280a0!important;background:linear-gradient(180deg,#0f1c2c,#0a1522)!important;box-shadow:0 8px 22px rgba(0,0,0,.11)!important}[data-testid="stFileUploaderDropzone"]:focus-within{outline:2px solid var(--accent)!important;outline-offset:3px!important}.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent-2))!important}[data-testid="stProgressBar"]{padding:3px!important;border:1px solid var(--line-soft)!important;border-radius:999px!important;background:#0a1420!important}.stExpander{overflow:hidden!important;border:1px solid var(--line)!important;border-radius:12px!important;background:#0b1522!important;box-shadow:var(--shadow-1)!important}.stExpander details{background:#0b1522!important}.stExpander summary{min-height:44px!important;padding:11px 14px!important}.stExpander summary:hover{background:rgba(18,31,48,.72)!important}.stExpander summary p{color:#dfe8f1!important;font-size:11px!important;font-weight:780!important}.stExpander details>div{padding:4px 14px 14px!important}.stAlert{border:1px solid var(--line)!important;border-radius:11px!important;background:#101b2a!important}

/* ─────────────── Research reading surfaces ─────────────── */
.answertext{max-width:920px!important;color:#e4ebf2!important;font-size:14px!important;line-height:1.84!important;letter-spacing:0!important;white-space:pre-wrap!important}.snippet{color:#a0afbe!important;font-size:10px!important;line-height:1.72!important}.code,pre{background:#09131f!important;border:1px solid var(--line-soft)!important;border-radius:11px!important}.empty{padding:38px 24px!important;border:1px dashed #334a64!important;border-radius:13px!important;color:#7b8ea4!important;font-size:10px!important;line-height:1.6!important}.page{gap:12px!important;padding:11px 0!important;border-bottom:1px solid var(--line-soft)!important;color:#9aabba!important;font-size:10px!important;line-height:1.45!important}.page small{margin-top:3px!important;color:#607389!important;font-size:9px!important}.timeline{margin:4px 0 0 5px!important;padding-left:16px!important;border-left:1px solid #2a3e56!important}.event{padding-bottom:15px!important;color:#92a4b8!important;font-size:10px!important;line-height:1.5!important}.event:before{left:-20px!important;top:4px!important;width:6px!important;height:6px!important;border-radius:50%!important;background:var(--accent)!important;box-shadow:0 0 0 4px #0d1622!important}.event b{color:#dce6ee!important}.event small{margin-top:3px!important;color:#60738b!important;font-size:9px!important}[data-testid="stDataFrame"]{overflow:hidden!important;border:1px solid var(--line-soft)!important;border-radius:12px!important;background:#0b1522!important;box-shadow:var(--shadow-1)!important}

/* ─────────────── Live state / feedback ─────────────── */
.livebar{min-height:42px!important;margin-top:12px!important;padding:10px 13px!important;gap:8px!important;border:1px solid var(--line-strong)!important;border-radius:12px!important;background:rgba(12,20,32,.92)!important;color:#9aacbf!important;font-size:10px!important}.livebar .grow{flex:1!important}.livepulse{width:7px!important;height:7px!important;flex:0 0 7px!important;border-radius:50%!important;background:var(--good)!important;box-shadow:0 0 0 5px rgba(94,230,161,.07),0 0 12px rgba(94,230,161,.55)!important}.livebar.state-processing .livepulse{background:var(--good)!important}.livebar.state-ready .livepulse{background:var(--accent-2)!important;box-shadow:0 0 0 5px rgba(130,152,255,.08),0 0 12px rgba(130,152,255,.55)!important}.livebar.state-idle .livepulse{background:#65758b!important;box-shadow:none!important}.livebar.state-ready{background:rgba(17,28,45,.88)!important}.livebar.state-idle{background:rgba(11,18,29,.9)!important;color:#7f90a5!important}

/* ─────────────── Links / menus / code ─────────────── */
a{color:#91b8ff!important;text-underline-offset:3px!important;text-decoration-color:rgba(145,184,255,.30)!important;transition:color .15s var(--ease),text-decoration-color .15s var(--ease)!important}a:hover{color:#bfd1ff!important;text-decoration-color:rgba(191,209,255,.72)!important}code{padding:1px 5px!important;border:1px solid var(--line-soft)!important;border-radius:6px!important;background:#111d2c!important;color:#b9d8dc!important}hr{margin:16px 0!important;border:0!important;border-top:1px solid var(--line-soft)!important}div[data-baseweb="popover"]>div,[role="dialog"]{border:1px solid var(--line-strong)!important;border-radius:14px!important;background:#0b1522!important;box-shadow:var(--shadow-3)!important}[role="menu"]{padding:7px!important;background:#0b1522!important}[role="menuitem"]{min-height:38px!important;padding:0 10px!important;border-radius:9px!important;color:#dce5ee!important}[role="menuitem"]:hover,[role="menuitem"]:focus{background:#132136!important;color:#fff!important}

/* ─────────────── Accessibility / motion ─────────────── */
button:focus-visible,input:focus-visible,textarea:focus-visible,[role="button"]:focus-visible,[role="tab"]:focus-visible,[role="option"]:focus-visible{outline:2px solid var(--accent)!important;outline-offset:3px!important}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important;animation:none!important}.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover,.stepcard:hover,.docrow:hover,.evidence:hover{transform:none!important}}

/* ─────────────── Responsive architecture ─────────────── */
@media(max-width:1180px){.block-container{padding-left:28px!important;padding-right:28px!important}.hero h1{font-size:34px!important}.statgrid,.grid4,.metricgrid{grid-template-columns:repeat(2,minmax(0,1fr))!important}}
@media(max-width:900px){[data-testid=stSidebar]{width:240px!important}.block-container{padding-left:22px!important;padding-right:22px!important}.twocol,.threecol{grid-template-columns:1fr!important}}
@media(max-width:760px){.block-container{padding:18px 12px 56px!important}.title{font-size:22px!important}.crumb{font-size:8px!important}.hero{padding:23px!important;border-radius:18px!important}.hero h1{font-size:28px!important;letter-spacing:-.04em!important}.hero p{font-size:11px!important;line-height:1.72!important}.hero-actions .stButton>button{min-height:48px!important}.section{padding:15px!important;border-radius:14px!important}.statgrid,.grid4,.metricgrid,.workflow{grid-template-columns:1fr!important}.microgrid{grid-template-columns:repeat(2,minmax(0,1fr))!important}.page{grid-template-columns:1fr!important;gap:4px!important}.page>div{min-width:0!important;overflow-wrap:anywhere!important}.top{align-items:flex-start!important}.topright{display:none!important}.livebar{align-items:flex-start!important;flex-wrap:wrap!important}.livebar .grow{display:none!important}.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:48px!important}.stTextInput input,.stNumberInput input,.stDateInput input,[data-baseweb="select"]>div{min-height:48px!important}.stTextArea textarea{min-height:116px!important}.stExpander summary{min-height:48px!important;padding:11px 12px!important}.stExpander details>div{padding:4px 12px 12px!important}.stFileUploader{padding:7px!important}[data-testid="stFileUploaderDropzone"]{min-height:96px!important}[data-testid="stForm"]{padding:10px!important;border-radius:12px!important}}
</style>'''


def _coherent_css() -> None:
    """Emit only the authoritative premium stylesheet; do not stack the legacy renovation CSS."""
    st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)


def _honest_live_indicator(system) -> None:
    """Distinguish actual processing from a ready or idle workspace."""
    active = len(ui.active_docs(system))
    ready = len(ui.ready_docs(system))
    running = ui._job_running()
    if active or running:
        state, label, cls = "PROCESSING", f"{active} document(s) actively processing", "state-processing"
    elif ready:
        state, label, cls = "READY", f"{ready} document(s) ready", "state-ready"
    else:
        state, label, cls = "IDLE", "Workspace idle", "state-idle"
    markup = f'<div class="livebar {cls}"><span class="livepulse"></span><b>{state}</b><span>{renovation.esc(label)}</span><span class="grow"></span>{renovation.status(state)}</div>'
    st.markdown(markup, unsafe_allow_html=True)


def _premium_command_palette() -> None:
    """Compact navigation search with an honest product label."""
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
    """Install the authoritative premium visual contract before UI security capture."""
    ui.css = _coherent_css
    renovation.live_indicator = _honest_live_indicator
    renovation.command_palette = _premium_command_palette
    ui.live_indicator = _honest_live_indicator
    ui.command_palette = _premium_command_palette


__all__ = ["apply"]
