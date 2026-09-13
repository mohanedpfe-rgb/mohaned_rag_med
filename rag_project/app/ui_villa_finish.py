from __future__ import annotations

import streamlit as st


_VILLA_CSS = r'''<style>
/* Final framework-to-product finish. Keep the existing BookRAG theme and remove residual Streamlit defaults. */
:root{
  --villa-surface:#0b1522;--villa-surface-2:#0f1a29;--villa-line:#263950;
  --villa-line-soft:#1a2a3d;--villa-text:#edf3f8;--villa-muted:#8295ab;
  --villa-accent:#70e1d6;--villa-accent-2:#8298ff;--villa-shadow:0 10px 30px rgba(0,0,0,.14);
}

/* Typography primitives */
.stCaption,.stMarkdown p{color:var(--villa-muted)!important}
.stMarkdown p{line-height:1.68!important}
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3{color:var(--villa-text)!important;letter-spacing:-.025em!important}
[data-testid="stMarkdownContainer"] h2{font-size:20px!important;line-height:1.2!important}
[data-testid="stMarkdownContainer"] h3{font-size:15px!important;line-height:1.25!important}

/* Popovers and menus: turn framework overlays into product surfaces. */
div[data-baseweb="popover"] > div{background:#0b1522!important;border:1px solid var(--villa-line)!important;border-radius:14px!important;box-shadow:0 24px 60px rgba(0,0,0,.34)!important}
[role="menu"]{background:#0b1522!important;border:0!important;padding:7px!important}
[role="menuitem"]{border-radius:9px!important;min-height:38px!important;color:#dce5ee!important}
[role="menuitem"]:hover{background:#132136!important}

/* Expanders */
[data-testid="stExpander"]{box-shadow:var(--villa-shadow)!important;overflow:hidden!important}
[data-testid="stExpander"] summary{padding:12px 14px!important}
[data-testid="stExpander"] summary:hover{background:rgba(18,31,48,.72)!important}
[data-testid="stExpander"] summary p{font-size:11px!important;font-weight:760!important;color:#dfe8f1!important}

/* Tabs / segmented controls */
.stTabs [data-baseweb="tab-list"]{gap:5px!important;padding:4px!important;background:#09131f!important;border:1px solid var(--villa-line-soft)!important;border-radius:12px!important}
.stTabs [data-baseweb="tab"]{min-height:36px!important;padding:0 13px!important;border-radius:9px!important;color:#8fa0b4!important;font-size:10px!important;font-weight:760!important}
.stTabs [aria-selected="true"]{background:#142237!important;color:#eef4f8!important;box-shadow:0 4px 12px rgba(0,0,0,.14)!important}
.stTabs [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--villa-accent),var(--villa-accent-2))!important;height:2px!important}

/* Select boxes / multiselect chips */
[data-baseweb="select"]>div{box-shadow:none!important}
[data-baseweb="tag"]{background:#16263b!important;border:1px solid #2f4662!important;color:#dce6ef!important;border-radius:8px!important}
[data-baseweb="tag"] span{font-size:9px!important}

/* Checkbox / radio */
[data-testid="stCheckbox"] label,[data-testid="stRadio"] label{color:#dbe5ee!important;font-size:10px!important}
[data-testid="stCheckbox"] [data-baseweb="checkbox"]{margin-right:7px!important}
[data-testid="stRadio"] [data-baseweb="radio"]{gap:7px!important}

/* Sliders */
[data-testid="stSlider"] [role="slider"]{box-shadow:0 0 0 3px rgba(112,225,214,.09)!important}
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child{height:4px!important}

/* Progress bars */
[data-testid="stProgressBar"]{padding:3px!important;background:#0a1420!important;border:1px solid var(--villa-line-soft)!important;border-radius:999px!important}
[data-testid="stProgressBar"] > div{border-radius:999px!important;overflow:hidden!important}

/* Dataframes / tables: readable, aligned, non-default surfaces. */
[data-testid="stDataFrame"]{border:1px solid var(--villa-line-soft)!important;border-radius:12px!important;overflow:hidden!important;box-shadow:var(--villa-shadow)!important}
[data-testid="stDataFrame"] iframe{background:#0b1522!important}

/* Toasts */
[data-testid="stToast"]{border:1px solid var(--villa-line)!important;border-radius:12px!important;background:#101c2b!important;box-shadow:0 18px 48px rgba(0,0,0,.30)!important}

/* Dividers */
hr{border:0!important;border-top:1px solid var(--villa-line-soft)!important;margin:15px 0!important}

/* Forms: remove excessive default vertical padding while preserving breathing room. */
[data-testid="stForm"]{border:1px solid var(--villa-line-soft)!important;border-radius:14px!important;background:rgba(10,20,32,.72)!important;padding:12px!important}

/* Upload drop zone micro-details. */
[data-testid="stFileUploaderDropzone"]{min-height:88px!important;border:1px dashed #3b5571!important;border-radius:10px!important;background:linear-gradient(180deg,#0c1725,#09131f)!important}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#5f7b9b!important;background:linear-gradient(180deg,#0f1c2c,#0a1522)!important}

/* Tooltips / help icons */
button[aria-label*="help"],button[title*="help"]{color:#768ba1!important}

/* Mobile optical cleanup */
@media(max-width:760px){
  .stTabs [data-baseweb="tab"]{min-height:42px!important;padding:0 11px!important}
  [data-baseweb="tag"]{min-height:26px!important}
  [data-testid="stExpander"] summary{padding:11px 12px!important}
  [data-testid="stForm"]{padding:10px!important}
}

@media(prefers-reduced-motion:reduce){
  [data-testid="stExpander"] summary,*{transition:none!important;animation:none!important}
}
</style>'''


def apply() -> None:
    """Apply only the final framework-surface finish; no runtime behavior is modified."""
    st.markdown(_VILLA_CSS, unsafe_allow_html=True)


__all__ = ["apply"]
