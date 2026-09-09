import streamlit as st

from rag_project.app.canva_exact_ui import main as exact_main


_EXACT_RUNTIME_CSS = """
<style>
/* 1920x1080 Canva reference calibration: shell 91,112,1738x855. */
.studio-shell{position:fixed!important;left:91px!important;top:112px!important;width:1738px!important;height:855px!important;z-index:0!important}
.block-container{position:relative!important;width:1441px!important;max-width:1441px!important;margin-left:342px!important;padding:23px 46px 56px!important}
.topline{position:fixed!important;left:388px!important;top:135px!important;width:1395px!important;height:45px!important;z-index:10!important;margin:0!important}
.title-block{position:fixed!important;left:388px!important;top:217px!important;width:760px!important;z-index:8!important;margin:0!important}
.index-grid{position:fixed!important;left:388px!important;top:360px!important;width:1395px!important;height:202px!important;z-index:7!important;margin:0!important}
.index-panel{height:202px!important;box-sizing:border-box!important}
.settings-panel{height:202px!important;box-sizing:border-box!important}
.row-grid{position:fixed!important;left:388px!important;top:600px!important;width:1375px!important;z-index:7!important;margin:0!important}
[data-testid="stSidebar"]{position:fixed!important;left:91px!important;top:112px!important;width:251px!important;height:855px!important;z-index:20!important}
[data-testid="stSidebar"] .brand{position:fixed!important;left:169px!important;top:154px!important;width:175px!important;height:45px!important;margin:0!important}
[data-testid="stSidebar"] .nav-label{display:none!important}
[class*="st-key-nav_Overview"],[class*="st-key-nav_Documents"],[class*="st-key-nav_Index them"],[class*="st-key-nav_Ingestion"],[class*="st-key-nav_Inspector"],[class*="st-key-nav_Settings"]{position:fixed!important;left:114px!important;width:252px!important;height:45px!important;z-index:25!important}
[class*="st-key-nav_Overview"]{top:202px!important}[class*="st-key-nav_Documents"]{top:270px!important}[class*="st-key-nav_Index them"]{top:315px!important}[class*="st-key-nav_Ingestion"]{top:382px!important}[class*="st-key-nav_Inspector"]{top:427px!important}[class*="st-key-nav_Settings"]{top:472px!important}
[class*="st-key-nav_secondary_Chat"],[class*="st-key-nav_secondary_Health"],[class*="st-key-nav_secondary_Background"]{position:fixed!important;left:114px!important;width:252px!important;height:40px!important;z-index:25!important}
[class*="st-key-nav_secondary_Chat"]{top:525px!important}[class*="st-key-nav_secondary_Health"]{top:565px!important}[class*="st-key-nav_secondary_Background"]{top:605px!important}
[class*="st-key-exact_overview_q"]{position:fixed!important;left:408px!important;top:650px!important;width:760px!important;z-index:15!important}
[class*="st-key-exact_overview_ask"]{position:fixed!important;left:408px!important;top:752px!important;width:190px!important;z-index:16!important}
[class*="st-key-exact_uploads"]{position:fixed!important;left:114px!important;top:660px!important;width:205px!important;height:48px!important;z-index:26!important;overflow:hidden!important}
[class*="st-key-exact_incoming"]{position:fixed!important;left:114px!important;top:713px!important;width:205px!important;z-index:26!important}
[class*="st-key-exact_start"]{position:fixed!important;left:114px!important;top:762px!important;width:205px!important;z-index:26!important}
[class*="st-key-exact_recreate"]{position:fixed!important;left:114px!important;top:811px!important;width:205px!important;z-index:26!important}
[class*="st-key-exact_confirm"]{position:fixed!important;left:114px!important;top:860px!important;width:205px!important;z-index:26!important}
[class*="st-key-exact_clear"]{position:fixed!important;left:114px!important;top:904px!important;width:205px!important;z-index:26!important}
</style>
"""


def main() -> None:
    exact_main()
    # Apply after rendering so CSS also targets Streamlit widgets by stable key classes.
    st.markdown(_EXACT_RUNTIME_CSS, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
