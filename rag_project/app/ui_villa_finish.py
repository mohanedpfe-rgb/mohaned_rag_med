from __future__ import annotations

import streamlit as st


_VILLA_CSS = r'''<style>
/* Final framework-to-product finish. Keep BookRAG behavior intact and remove residual framework defaults. */
:root{
  --villa-surface:#0b1522;--villa-surface-2:#0f1a29;--villa-line:#263950;
  --villa-line-soft:#1a2a3d;--villa-line-strong:#344c68;--villa-text:#edf3f8;
  --villa-muted:#8295ab;--villa-faint:#667b93;--villa-accent:#70e1d6;--villa-accent-2:#8298ff;
  --villa-shadow:0 10px 30px rgba(0,0,0,.14);--villa-shadow-lg:0 24px 60px rgba(0,0,0,.34);
  --villa-ease:cubic-bezier(.22,.8,.2,1);
}

/* Typography primitives */
.stCaption{font-size:10px!important;line-height:1.55!important;color:var(--villa-muted)!important}
.stMarkdown p{font-size:11px!important;line-height:1.68!important;color:var(--villa-muted)!important}
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3{color:var(--villa-text)!important;letter-spacing:-.025em!important}
[data-testid="stMarkdownContainer"] h1{font-size:28px!important;line-height:1.12!important}
[data-testid="stMarkdownContainer"] h2{font-size:20px!important;line-height:1.2!important}
[data-testid="stMarkdownContainer"] h3{font-size:15px!important;line-height:1.25!important}

/* Framework rhythm: remove accidental vertical drift without making pages cramped. */
[data-testid="stVerticalBlock"] > [data-testid="element-container"] + [data-testid="element-container"]{margin-top:2px!important}
[data-testid="stHorizontalBlock"]{gap:10px!important}
.stCaption + [data-testid="stMarkdownContainer"]{margin-top:0!important}

/* Popovers and menus: high-elevation product surfaces. */
div[data-baseweb="popover"] > div{background:#0b1522!important;border:1px solid var(--villa-line-strong)!important;border-radius:14px!important;box-shadow:var(--villa-shadow-lg)!important}
[role="menu"]{background:#0b1522!important;border:0!important;padding:7px!important}
[role="menuitem"]{border-radius:9px!important;min-height:38px!important;padding:0 10px!important;color:#dce5ee!important;transition:background .15s var(--villa-ease),color .15s var(--villa-ease)!important}
[role="menuitem"]:hover,[role="menuitem"]:focus{background:#132136!important;color:#fff!important}

/* Expanders: compact header, calm body, deliberate disclosure affordance. */
[data-testid="stExpander"]{box-shadow:var(--villa-shadow)!important;overflow:hidden!important;border-color:var(--villa-line)!important}
[data-testid="stExpander"] summary{min-height:42px!important;padding:11px 14px!important}
[data-testid="stExpander"] summary:hover{background:rgba(18,31,48,.72)!important}
[data-testid="stExpander"] summary p{font-size:11px!important;font-weight:760!important;color:#dfe8f1!important}
[data-testid="stExpander"] details > div{padding:4px 14px 14px!important}

/* Tabs / segmented controls */
.stTabs [data-baseweb="tab-list"]{gap:5px!important;padding:4px!important;background:#09131f!important;border:1px solid var(--villa-line-soft)!important;border-radius:12px!important}
.stTabs [data-baseweb="tab"]{min-height:36px!important;padding:0 13px!important;border-radius:9px!important;color:#8fa0b4!important;font-size:10px!important;font-weight:760!important;transition:background .15s var(--villa-ease),color .15s var(--villa-ease),transform .15s var(--villa-ease)!important}
.stTabs [data-baseweb="tab"]:hover{background:#101e30!important;color:#dfe8f1!important}
.stTabs [aria-selected="true"]{background:#142237!important;color:#eef4f8!important;box-shadow:0 4px 12px rgba(0,0,0,.14)!important}
.stTabs [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--villa-accent),var(--villa-accent-2))!important;height:2px!important}

/* Select boxes / multiselect chips / dropdown menus */
[data-baseweb="select"]>div{box-shadow:none!important}
[data-baseweb="tag"]{background:#16263b!important;border:1px solid #2f4662!important;color:#dce6ef!important;border-radius:8px!important;min-height:24px!important}
[data-baseweb="tag"] span{font-size:9px!important}
[role="listbox"]{background:#0b1522!important;border:1px solid var(--villa-line)!important;border-radius:12px!important;box-shadow:var(--villa-shadow-lg)!important;padding:5px!important}
[role="option"]{border-radius:8px!important;color:#d9e3ed!important;min-height:34px!important;padding:7px 9px!important}
[role="option"]:hover,[role="option"][aria-selected="true"]{background:#142237!important;color:#fff!important}

/* Checkbox / radio */
[data-testid="stCheckbox"] label,[data-testid="stRadio"] label{color:#dbe5ee!important;font-size:10px!important;line-height:1.45!important}
[data-testid="stCheckbox"] [data-baseweb="checkbox"]{margin-right:7px!important}
[data-testid="stRadio"] [data-baseweb="radio"]{gap:7px!important}

/* Sliders */
[data-testid="stSlider"] [role="slider"]{box-shadow:0 0 0 3px rgba(112,225,214,.09)!important;transition:box-shadow .15s var(--villa-ease)!important}
[data-testid="stSlider"] [role="slider"]:focus-visible{outline:2px solid var(--villa-accent)!important;outline-offset:3px!important}
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child{height:4px!important}

/* Progress bars */
[data-testid="stProgressBar"]{padding:3px!important;background:#0a1420!important;border:1px solid var(--villa-line-soft)!important;border-radius:999px!important}
[data-testid="stProgressBar"] > div{border-radius:999px!important;overflow:hidden!important}

/* Dataframes / tables: readable, aligned, non-default surfaces. */
[data-testid="stDataFrame"]{border:1px solid var(--villa-line-soft)!important;border-radius:12px!important;overflow:hidden!important;box-shadow:var(--villa-shadow)!important;background:#0b1522!important}
[data-testid="stDataFrame"] iframe{background:#0b1522!important}
[data-testid="stDataFrame"] button{border-radius:8px!important}
[data-testid="stDataFrame"] [role="columnheader"]{background:#101d2d!important;color:#9fb0c3!important}

/* Toasts / dialogs */
[data-testid="stToast"]{border:1px solid var(--villa-line)!important;border-radius:12px!important;background:#101c2b!important;box-shadow:0 18px 48px rgba(0,0,0,.30)!important}
[role="dialog"]{background:#0b1522!important;border:1px solid var(--villa-line)!important;border-radius:14px!important;box-shadow:var(--villa-shadow-lg)!important}

/* Dividers */
hr{border:0!important;border-top:1px solid var(--villa-line-soft)!important;margin:15px 0!important}

/* Forms: consistent panel treatment. */
[data-testid="stForm"]{border:1px solid var(--villa-line-soft)!important;border-radius:14px!important;background:rgba(10,20,32,.72)!important;padding:12px!important;box-shadow:0 7px 22px rgba(0,0,0,.08)!important}
[data-testid="stFormSubmitButton"]{margin-top:3px!important}

/* Upload drop zone micro-details. */
[data-testid="stFileUploaderDropzone"]{min-height:88px!important;border:1px dashed #3b5571!important;border-radius:10px!important;background:linear-gradient(180deg,#0c1725,#09131f)!important;transition:border-color .15s var(--villa-ease),background .15s var(--villa-ease),box-shadow .15s var(--villa-ease)!important}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#5f7b9b!important;background:linear-gradient(180deg,#0f1c2c,#0a1522)!important;box-shadow:0 6px 18px rgba(0,0,0,.10)!important}
[data-testid="stFileUploaderDropzone"]:focus-within{outline:2px solid var(--villa-accent)!important;outline-offset:3px!important}

/* Tooltips / help icons */
button[aria-label*="help"],button[title*="help"]{color:#768ba1!important}

/* Inline code / code blocks */
code{background:#111d2c!important;border:1px solid var(--villa-line-soft)!important;border-radius:6px!important;padding:1px 5px!important;color:#b9d8dc!important}
pre{background:#09131f!important;border:1px solid var(--villa-line-soft)!important;border-radius:11px!important;box-shadow:var(--villa-shadow)!important}

/* Links: understated until interacted with. */
a{color:#8fb8ff!important;text-decoration-color:rgba(143,184,255,.30)!important;text-underline-offset:3px!important}
a:hover{color:#b8ccff!important;text-decoration-color:rgba(184,204,255,.72)!important}

/* Micro accessibility: keyboard visibility is deliberately stronger than hover. */
button:focus-visible,input:focus-visible,textarea:focus-visible,[role="button"]:focus-visible,[role="tab"]:focus-visible,[role="option"]:focus-visible{outline:2px solid var(--villa-accent)!important;outline-offset:3px!important}

/* Mobile optical cleanup */
@media(max-width:760px){
  .stTabs [data-baseweb="tab"]{min-height:42px!important;padding:0 11px!important}
  [data-baseweb="tag"]{min-height:26px!important}
  [data-testid="stExpander"] summary{min-height:46px!important;padding:11px 12px!important}
  [data-testid="stExpander"] details > div{padding:4px 12px 12px!important}
  [data-testid="stForm"]{padding:10px!important;border-radius:12px!important}
  div[data-baseweb="popover"] > div{max-width:calc(100vw - 24px)!important}
  [role="listbox"]{max-width:calc(100vw - 24px)!important}
  pre{overflow-x:auto!important}
}

@media(prefers-reduced-motion:reduce){
  *,*::before,*::after{scroll-behavior:auto!important;transition:none!important;animation:none!important}
}
</style>'''


def apply() -> None:
    """Apply only the final framework-surface finish; no runtime behavior is modified."""
    st.markdown(_VILLA_CSS, unsafe_allow_html=True)


__all__ = ["apply"]
