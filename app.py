import streamlit as st

from rag_project.app.canva_exact_ui import main as exact_main


_EXACT_RUNTIME_CSS = """
<style>
/* Fixed 1920x1080 calibration from Canva reference. */
.studio-shell{position:fixed!important;left:91px!important;top:112px!important;width:1738px!important;height:855px!important;z-index:0!important}
.block-container{position:relative!important;width:1441px!important;max-width:1441px!important;margin-left:342px!important;padding:23px 46px 56px!important}
.topline{position:fixed!important;left:388px!important;top:135px!important;width:1395px!important;height:45px!important;z-index:10!important;margin:0!important}
.title-block{position:fixed!important;left:388px!important;top:217px!important;width:760px!important;z-index:8!important;margin:0!important}
.index-grid{position:fixed!important;left:388px!important;top:360px!important;width:1395px!important;height:202px!important;z-index:7!important;margin:0!important}
.index-panel{position:fixed!important;left:388px!important;top:360px!important;width:802px!important;height:202px!important;box-sizing:border-box!important}
.settings-panel{position:fixed!important;left:1210px!important;top:360px!important;width:573px!important;height:202px!important;box-sizing:border-box!important}
.index-metric{position:fixed!important;top:405px!important;height:158px!important;box-sizing:border-box!important;z-index:8!important}
[data-testid="stMarkdownContainer"] .index-metric:nth-of-type(1){left:731px!important;width:161px!important}
[data-testid="stMarkdownContainer"] .index-metric:nth-of-type(2){left:868px!important;width:162px!important}
[data-testid="stMarkdownContainer"] .index-metric:nth-of-type(3){left:1006px!important;width:161px!important}
.query-panel{position:fixed!important;left:388px!important;top:600px!important;width:802px!important;height:74px!important;box-sizing:border-box!important;z-index:7!important;overflow:visible!important}
.health-panel{position:fixed!important;left:388px!important;top:694px!important;width:802px!important;height:236px!important;box-sizing:border-box!important;z-index:7!important}
.inspector-panel{position:fixed!important;left:1210px!important;top:694px!important;width:573px!important;height:236px!important;box-sizing:border-box!important;z-index:8!important}
.query-panel .panel-title{margin-top:0!important}.health-panel .panel-head{margin-bottom:18px!important}.inspector-panel .panel-head{margin-bottom:18px!important}
[class*="st-key-exact_overview_q"]{position:fixed!important;left:408px!important;top:642px!important;width:760px!important;z-index:15!important}
[class*="st-key-exact_overview_ask"]{position:fixed!important;left:408px!important;top:748px!important;width:190px!important;z-index:16!important}
[data-testid="stSidebar"]{position:fixed!important;left:91px!important;top:112px!important;width:251px!important;height:855px!important;z-index:20!important}
[data-testid="stSidebar"] .brand{position:fixed!important;left:169px!important;top:154px!important;width:175px!important;height:45px!important;margin:0!important}
[data-testid="stSidebar"] .nav-label{display:none!important}
[class*="st-key-nav_Overview"],[class*="st-key-nav_Documents"],[class*="st-key-nav_Index them"],[class*="st-key-nav_Ingestion"],[class*="st-key-nav_Inspector"],[class*="st-key-nav_Settings"]{position:fixed!important;left:114px!important;width:252px!important;height:45px!important;z-index:25!important}
[class*="st-key-nav_Overview"]{top:202px!important}[class*="st-key-nav_Documents"]{top:270px!important}[class*="st-key-nav_Index them"]{top:315px!important}[class*="st-key-nav_Ingestion"]{top:382px!important}[class*="st-key-nav_Inspector"]{top:427px!important}[class*="st-key-nav_Settings"]{top:472px!important}
[class*="st-key-nav_secondary_Chat"],[class*="st-key-nav_secondary_Health"],[class*="st-key-nav_secondary_Background"]{position:fixed!important;left:114px!important;width:205px!important;height:38px!important;z-index:25!important}
[class*="st-key-nav_secondary_Chat"]{top:525px!important}[class*="st-key-nav_secondary_Health"]{top:565px!important}[class*="st-key-nav_secondary_Background"]{top:605px!important}
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
    st.markdown(_EXACT_RUNTIME_CSS, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
