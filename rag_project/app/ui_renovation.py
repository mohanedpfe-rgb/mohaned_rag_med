from __future__ import annotations

import hashlib
import html
import json
from typing import Any

import streamlit as st

from rag_project.app import bookrag_ui as ui


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def status(value: Any) -> str:
    return ui._status(value)


def inject_css() -> None:
    st.markdown(
        r'''<style>
:root{--bg:#070b12;--surface:#0f1724;--surface2:#121d2d;--line:#223148;--line2:#18253a;--text:#f4f7fb;--muted:#8b9bb0;--faint:#5d6e84;--accent:#70e1d6;--accent2:#8298ff;--good:#5ee6a1;--warn:#f2c96d;--bad:#ff788b;--shadow:0 18px 50px rgba(0,0,0,.28)}
html,body,[class*=css]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.stApp{background:radial-gradient(900px 560px at 92% -5%,rgba(130,152,255,.14),transparent 64%),radial-gradient(720px 480px at 3% 0,rgba(112,225,214,.08),transparent 60%),var(--bg);color:var(--text)}
[data-testid=stHeader]{height:0;background:transparent}.block-container{max-width:1480px;padding:28px 36px 84px}
[data-testid=stSidebar]{background:linear-gradient(180deg,#0a111c,#08101a)!important;border-right:1px solid var(--line)!important}[data-testid=stSidebar]>div:first-child{padding:20px 14px 26px!important}
.brand{display:flex;gap:12px;align-items:center;padding:5px 8px 24px}.brandmark{width:40px;height:40px;border-radius:13px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;color:#061017;font-size:12px;font-weight:950;box-shadow:0 12px 34px rgba(112,225,214,.18)}.brandname{font-weight:900;font-size:15px}.brandsub{font-size:9px;color:var(--faint);margin-top:3px}.navgroup{margin:16px 8px 7px;color:#5f7087;font-size:9px;text-transform:uppercase;letter-spacing:.15em;font-weight:900}
[data-testid=stSidebar] .stButton>button{min-height:42px!important;border:1px solid transparent!important;background:transparent!important;color:#93a3b7!important;border-radius:10px!important;text-align:left!important;font-size:11px!important;font-weight:750!important;padding:0 12px!important}[data-testid=stSidebar] .stButton>button:hover{background:#111c2b!important;border-color:#26374e!important;color:#f4f7fb!important}.sidefoot{margin-top:18px;padding:14px 8px;border-top:1px solid var(--line2);color:#68798f;font-size:9px;line-height:1.65}
.top{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-bottom:15px}.crumb{font-size:9px;color:#607188;text-transform:uppercase;letter-spacing:.14em;font-weight:900}.title{font-size:24px;font-weight:900;letter-spacing:-.04em;margin-top:4px}.topright{display:flex;gap:8px}.chip{height:31px;padding:0 10px;border-radius:999px;border:1px solid var(--line);background:rgba(15,23,36,.84);display:flex;align-items:center;font-size:9px;color:#9bacbf}.dot{width:6px;height:6px;border-radius:50%;background:var(--good);box-shadow:0 0 11px rgba(94,230,161,.8);margin-right:7px}
.hero{position:relative;overflow:hidden;border:1px solid #293b55;border-radius:22px;padding:30px;background:linear-gradient(125deg,#0d1624,#102035 56%,#162d37);box-shadow:var(--shadow)}.hero:after{content:"";position:absolute;right:-120px;top:-170px;width:380px;height:380px;border-radius:50%;background:radial-gradient(circle,rgba(112,225,214,.15),transparent 68%)}.hero .kicker{position:relative;z-index:1;font-size:9px;text-transform:uppercase;letter-spacing:.17em;color:var(--accent);font-weight:900}.hero h1{position:relative;z-index:1;font-size:36px;line-height:1.05;letter-spacing:-.05em;margin:9px 0}.hero p{position:relative;z-index:1;max-width:760px;color:#9aaabd;line-height:1.7;font-size:12px;margin:0}.hero-actions{position:relative;z-index:1;margin-top:20px}
.section{border:1px solid var(--line);border-radius:16px;background:rgba(15,23,36,.9);padding:19px;margin-top:13px;box-shadow:0 10px 34px rgba(0,0,0,.10)}.sectionhead{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;margin-bottom:13px}.sectiontitle{font-size:14px;font-weight:900;letter-spacing:-.02em}.sectionsub{font-size:10px;color:#6f8198;line-height:1.6;margin-top:4px}
.statgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:13px}.stat{border:1px solid var(--line);border-radius:14px;background:linear-gradient(180deg,rgba(18,29,45,.92),rgba(12,19,30,.92));padding:15px;min-height:102px}.statlabel{font-size:9px;color:#6f8198;text-transform:uppercase;letter-spacing:.09em;font-weight:850}.statvalue{font-size:28px;font-weight:900;letter-spacing:-.055em;margin-top:6px}.stathint{font-size:9px;color:#56677d;margin-top:3px}
.twocol{display:grid;grid-template-columns:minmax(0,1.32fr) minmax(300px,.68fr);gap:13px}.workflow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.stepcard{border:1px solid var(--line2);background:#0b1421;border-radius:12px;padding:13px}.stepnum{font-size:9px;color:var(--accent);font-weight:900}.stepname{font-size:11px;font-weight:850;margin-top:4px}.stepdesc{font-size:9px;line-height:1.6;color:#77889e;margin-top:4px}
.docrow,.evidence,.metric{border:1px solid var(--line2);background:#0b1421;border-radius:12px}.docrow{padding:13px;margin-top:9px}.dochead,.evidencehead{display:flex;justify-content:space-between;gap:12px}.docname{font-size:12px;font-weight:850;overflow-wrap:anywhere}.meta,.evidencemeta{font-size:9px;color:#6c7e95;margin-top:3px}.meter{height:5px;background:#19283a;border-radius:99px;overflow:hidden;margin:11px 0 8px}.meter i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:99px}.microgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.micro{padding:8px;border-radius:9px;border:1px solid #1b2b40;background:#0d1725}.micro span{display:block;font-size:8px;color:#60728a;text-transform:uppercase}.micro b{display:block;font-size:10px;margin-top:3px;overflow-wrap:anywhere}.pill{font-size:8px!important;padding:4px 8px!important}
.metricgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:12px}.metric{padding:12px}.metric label{display:block;font-size:8px;color:#60728a;text-transform:uppercase;letter-spacing:.07em}.metric b{display:block;font-size:15px;margin-top:5px}.metric small{display:block;font-size:8px;color:#586a80;margin-top:3px}.page{display:grid;grid-template-columns:1.05fr .7fr 1fr .65fr;gap:10px;padding:10px 0;border-bottom:1px solid #18263a;align-items:center;font-size:9px}.page:last-child{border-bottom:0}.page small{display:block;color:#56697f;font-size:8px;margin-top:2px}.empty{border:1px dashed #33465f;border-radius:12px;padding:34px;text-align:center;color:#75869b;font-size:10px}.timeline{border-left:1px solid #2a3b54;margin:4px 0 0 5px;padding-left:15px}.event{position:relative;padding:0 0 14px;color:#91a2b6;font-size:9px}.event:before{content:"";position:absolute;left:-19px;top:3px;width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px #0f1724}.event b{color:#d9e3ed}.event small{display:block;color:#596b82;margin-top:3px}
.livebar{display:flex;align-items:center;gap:8px;margin-top:12px;padding:10px 12px;border:1px solid var(--line);border-radius:11px;background:rgba(12,20,32,.9);font-size:10px;color:#8fa1b6}.grow{flex:1}.livepulse{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 0 5px rgba(94,230,161,.07),0 0 12px rgba(94,230,161,.55)}
.answertext{font-size:14px;line-height:1.8;color:#e1e8f0;white-space:pre-wrap}.snippet{font-size:10px;color:#9baabc;line-height:1.65;margin-top:7px}
.stButton>button{min-height:39px!important;border-radius:10px!important;border:1px solid #2b3d55!important;background:#111d2c!important;color:#e5edf5!important;font-weight:780!important;font-size:10px!important}.stButton>button:hover{background:#16253a!important;border-color:#4d657f!important}.stButton>button:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px}.stButton>button[kind=primary]{background:linear-gradient(135deg,#4d8796,#516ea8)!important;border-color:#6a9eb2!important;color:#fff!important}.stTextInput input,.stTextArea textarea,.stNumberInput input,[data-baseweb=select]{background:#0b1421!important;color:#edf3fa!important;border-color:#293b53!important;border-radius:10px!important}.stTextInput input:focus,.stTextArea textarea:focus{border-color:#5f9ea3!important;box-shadow:0 0 0 1px #5f9ea3!important}.stFileUploader{border:1px dashed #3a526e!important;border-radius:12px!important;background:#0b1421!important}.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important}.stExpander{border:1px solid #263950!important;border-radius:11px!important;background:#0b1421!important}.stAlert{border-radius:10px!important}
@media(max-width:1150px){.statgrid{grid-template-columns:repeat(2,1fr)}.twocol{grid-template-columns:1fr}.metricgrid{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){.block-container{padding:18px 13px 52px}.hero h1{font-size:27px}.statgrid,.metricgrid,.workflow{grid-template-columns:1fr}.microgrid{grid-template-columns:repeat(2,1fr)}.page{grid-template-columns:1fr 1fr}.topright{display:none}}
</style>''', unsafe_allow_html=True)


def command_palette() -> None:
    with st.popover("⌘K  Quick navigation"):
        query = st.text_input("Find a page", placeholder="Home, Documents, Ask…", key="renovation_command")
        normalized = (query or "").strip().casefold()
        commands = {"home":"Home","documents":"Documents","processing":"Live Processing","ask":"Ask BookRAG","inspector":"Inspector","system":"System","settings":"Settings"}
        for key, page in commands.items():
            if normalized and normalized not in key and normalized not in page.casefold():
                continue
            if st.button(f"{page}  →", key=f"renovation_command_{key}", use_container_width=True):
                ui._navigate(page)


def sidebar(system) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brandmark">BR</div><div><div class="brandname">BookRAG Medical</div><div class="brandsub">Evidence-first research workspace</div></div></div>', unsafe_allow_html=True)
        current = st.session_state.get("bookrag_page", "Home")
        for group, items in [("Workspace",["Home","Documents","Live Processing","Ask BookRAG"]),("Research tools",["Inspector","System","Settings"])]:
            st.markdown(f'<div class="navgroup">{group}</div>', unsafe_allow_html=True)
            for item in items:
                marker = "●" if item == current else "○"
                if st.button(f"{marker}  {item}", key=f"renovation_nav_{item}", use_container_width=True):
                    ui._navigate(item)
        st.markdown('<div class="sidefoot"><b>Local & private</b><br>Documents, retrieval and generation remain inside the configured local runtime.</div>', unsafe_allow_html=True)


def topbar(system, title: str) -> None:
    active = len(ui.active_docs(system))
    st.markdown(f'<div class="top"><div><div class="crumb">BookRAG · Research workspace</div><div class="title">{esc(title)}</div></div><div class="topright"><div class="chip"><span class="dot"></span>{active} processing</div></div></div>', unsafe_allow_html=True)
    command_palette()


@st.fragment(run_every="2s")
def live_indicator(system) -> None:
    active = len(ui.active_docs(system)); ready = len(ui.ready_docs(system))
    state = "PROCESSING" if active or ui._job_running() else ("READY" if ready else "IDLE")
    label = f"{active} document(s) actively processing" if active else (f"{ready} document(s) ready" if ready else "Workspace idle")
    st.markdown(f'<div class="livebar"><span class="livepulse"></span><b>LIVE</b><span>{esc(label)}</span><span class="grow"></span>{status(state)}</div>', unsafe_allow_html=True)


def upload_block(system) -> None:
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Build your library</div><div class="sectionsub">Add PDFs once. The existing ingestion pipeline handles validation, extraction, OCR, chunking, embeddings and publication.</div></div></div>', unsafe_allow_html=True)
    files = st.file_uploader("PDF documents", type=["pdf"], accept_multiple_files=True, key="renovation_uploader", label_visibility="collapsed")
    if files:
        seen = st.session_state.setdefault("uploaded_hashes", set()); added = 0; errors = []
        for file in files:
            payload = file.getvalue(); digest = hashlib.sha256(payload).hexdigest()
            if digest in seen: continue
            try:
                ui.save_pdf(ui.Path(system.settings.incoming_dir), file.name, payload); seen.add(digest); added += 1
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")
        for error in errors: st.error(error)
        if added:
            job_id = ui.auto_ingest(system, added)
            if job_id:
                st.session_state["bookrag_page"] = "Live Processing"; st.success(f"{added} PDF(s) accepted · live processing started"); st.rerun()
            elif st.session_state.get("auto_ingest_error"):
                st.error(st.session_state["auto_ingest_error"])
    st.markdown('</div>', unsafe_allow_html=True)


def home(system) -> None:
    topbar(system, "Research workspace")
    st.markdown('<div class="hero"><div class="kicker">Private medical document intelligence</div><h1>Find evidence. Understand it. Cite it.</h1><p>BookRAG turns medical PDFs into a searchable local knowledge base, makes processing visible, and answers only when retrieved evidence can support the response.</p><div class="hero-actions">', unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        if st.button("＋  Add research PDFs", key="renovation_home_add", type="primary", use_container_width=True): ui._navigate("Documents")
    with b:
        if st.button("Ask the library  →", key="renovation_home_ask", use_container_width=True): ui._navigate("Ask BookRAG")
    st.markdown('</div></div>', unsafe_allow_html=True)
    live_indicator(system)
    documents, ready, active = ui.docs(system), ui.ready_docs(system), ui.active_docs(system)
    stats = [("Documents",len(documents),"managed PDFs"),("Ready",len(ready),"published to retrieval"),("Processing",len(active),"live pipeline jobs"),("Search sections",ui.total_chunks(system),"indexed retrieval units")]
    st.markdown('<div class="statgrid">'+''.join(f'<div class="stat"><div class="statlabel">{esc(a)}</div><div class="statvalue">{b:,}</div><div class="stathint">{esc(c)}</div></div>' for a,b,c in stats)+'</div>', unsafe_allow_html=True)
    left,right = st.columns([1.32,.68])
    with left:
        st.markdown('<div class="section"><div class="sectiontitle">From PDF to defensible answer</div><div class="sectionsub">The interface follows the research workflow instead of exposing implementation details first.</div><div class="workflow">', unsafe_allow_html=True)
        steps = [("01","Add documents","Upload PDFs and start the existing protected ingestion pipeline.","Documents"),("02","Watch processing","See durable page state, OCR, stages and current progress.","Live Processing"),("03","Inspect evidence","Audit pages, events, chunks and index health.","Inspector"),("04","Ask grounded questions","Retrieve evidence first and abstain when support is insufficient.","Ask BookRAG")]
        for num,name,desc,target in steps:
            st.markdown(f'<div class="stepcard"><div class="stepnum">{num}</div><div class="stepname">{esc(name)}</div><div class="stepdesc">{esc(desc)}</div></div>', unsafe_allow_html=True)
            if st.button(f"Open {name} →", key=f"renovation_home_{num}"): ui._navigate(target)
        st.markdown('</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><div class="sectiontitle">Trust gates</div><div class="sectionsub">What protects you before an answer reaches you.</div>', unsafe_allow_html=True)
        for name,desc in [("Retrieve","Find relevant source evidence."),("Rerank","Prioritize the strongest passages."),("Ground","Constrain generation to retrieved evidence."),("Verify","Check grounding, citations and consistency."),("Abstain","Stop instead of inventing support.")]:
            st.markdown(f'<div class="evidence"><div class="evidencetitle">{esc(name)}</div><div class="snippet">{esc(desc)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def documents_page(system) -> None:
    topbar(system,"Documents"); upload_block(system); all_docs = ui.docs(system)
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Document library</div><div class="sectionsub">Filter, select and inspect persistent ingestion state.</div></div></div>', unsafe_allow_html=True)
    rows = ui._filter_documents(all_docs); ids = [str(d.get("document_id")) for d in rows if d.get("document_id")]
    selected = {x for x in st.session_state.setdefault("selected_docs",set()) if x in ids}
    select_all = st.checkbox(f"Select all visible · {len(rows)}", value=bool(rows) and len(selected)==len(ids), key="renovation_select_all")
    if select_all: selected=set(ids)
    st.session_state["selected_docs"] = selected
    if selected:
        a,b,c = st.columns([1,1,2])
        with a:
            if st.button("Verify selected",type="primary",use_container_width=True,key="renovation_verify_selected"):
                result={}
                for doc_id in sorted(selected):
                    try: result[doc_id]=system.verify_index(doc_id)
                    except Exception as exc: result[doc_id]={"error":str(exc)}
                st.session_state["bulk_verify_result"]=result
        with b:
            if st.button("Clear selection",use_container_width=True,key="renovation_clear_selection"):
                st.session_state["selected_docs"]=set(); st.rerun()
        with c:
            manifest=[{"document_id":d.get("document_id"),"file_name":ui._doc_name(d),"status":d.get("status"),"pages":d.get("total_pages"),"chunks":d.get("chunk_count"),"embeddings":d.get("embedding_count")} for d in rows if str(d.get("document_id")) in selected]
            st.download_button("Export selection manifest",data=json.dumps(manifest,indent=2).encode(),file_name="bookrag_selection.json",mime="application/json",use_container_width=True)
    if st.session_state.get("bulk_verify_result"):
        with st.expander("Verification results",expanded=False): st.json(st.session_state.pop("bulk_verify_result"))
    if not rows: st.markdown('<div class="empty">No documents match the current filters.</div>',unsafe_allow_html=True)
    for document in rows:
        doc_id=str(document.get("document_id") or ""); name=ui._doc_name(document); progress=ui._progress(document); checked=doc_id in selected
        st.markdown(f'<div class="docrow"><div class="dochead"><div><div class="docname">{esc(name)}</div><div class="meta">{esc(doc_id)} · {ui._fmt_bytes(document.get("file_size"))} · {esc((document.get("language") or "language unknown").upper())}</div></div>{status(document.get("status"))}</div><div class="meter"><i style="width:{progress*100:.1f}%"></i></div><div class="microgrid"><div class="micro"><span>Stage</span><b>{esc(ui._stage(document)[0])}</b></div><div class="micro"><span>Pages</span><b>{ui._int(document.get("current_page"))} / {ui._int(document.get("total_pages"))}</b></div><div class="micro"><span>Chunks</span><b>{ui._int(document.get("chunk_count")):,}</b></div><div class="micro"><span>Embeddings</span><b>{ui._int(document.get("embedding_count")):,}</b></div></div></div>',unsafe_allow_html=True)
        a,b=st.columns(2)
        with a:
            picked=st.checkbox("Select",value=checked,key=f"renovation_select_{doc_id}",label_visibility="collapsed")
            if picked!=checked:
                if picked:selected.add(doc_id)
                else:selected.discard(doc_id)
                st.session_state["selected_docs"]=selected; st.rerun()
        with b:
            if st.button("Inspect document →",key=f"renovation_inspect_{doc_id}",use_container_width=True):
                st.session_state["inspect_doc_id"]=doc_id; ui._navigate("Inspector")
        ui._document_preview(system,document)
    st.markdown('</div>',unsafe_allow_html=True)


@st.fragment(run_every="2s")
def processing_page(system) -> None:
    topbar(system,"Live Processing"); active=ui.active_docs(system)
    if not active:
        st.markdown('<div class="section"><div class="empty">No document is processing right now. Add a PDF from Documents to start.</div></div>',unsafe_allow_html=True); return
    latest={}
    for event in ui.events(system,limit=800): latest[str(event.get("document_id"))]=event
    stage_keys=list(ui.STAGES.keys())[:10]
    for document in active:
        doc_id=str(document.get("document_id")); stage,description=ui._stage(document); progress=ui._progress(document); current_stage=str(document.get("current_stage") or document.get("status") or "RUNNING").upper(); idx=stage_keys.index(current_stage) if current_stage in stage_keys else 0
        st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">{esc(ui._doc_name(document))}</div><div class="sectionsub">{esc(description)}</div></div>{status(document.get("status"))}</div>',unsafe_allow_html=True)
        st.progress(progress,text=f"Pipeline progress · {progress*100:.0f}%")
        latest_event=latest.get(doc_id,{})
        metrics=[("Current page",f'{ui._int(document.get("current_page")) or "—"} / {ui._int(document.get("total_pages")) or "—"}',"durable checkpoint"),("Stage",stage,"persisted state"),("Latest event",latest_event.get("message") or latest_event.get("event_type") or "Working…","most recent signal"),("Elapsed",ui._fmt_time(ui._elapsed(document.get("ingestion_started_at"))),"live wall-clock time")]
        st.markdown('<div class="metricgrid">'+''.join(f'<div class="metric"><label>{esc(k)}</label><b>{esc(v)}</b><small>{esc(h)}</small></div>' for k,v,h in metrics)+'</div>',unsafe_allow_html=True)
        blocks=[]
        for i in range(len(stage_keys)):
            style="background:linear-gradient(90deg,#70e1d6,#8298ff)" if i<idx else "background:#1a2a3d"
            shadow=";box-shadow:0 0 0 2px rgba(112,225,214,.12)" if i==idx else ""
            blocks.append(f'<div style="height:6px;flex:1;border-radius:99px;{style}{shadow}"></div>')
        st.markdown('<div style="display:flex;gap:5px;margin-top:12px">'+''.join(blocks)+'</div>',unsafe_allow_html=True)
        page_rows=ui.pages(system,doc_id)
        if page_rows:
            done=sum(1 for row in page_rows if str(row.get("extraction_status","")).upper()=="COMPLETED"); st.caption(f"{done} / {len(page_rows)} page records completed")
            with st.expander(f"Page-level extraction · {len(page_rows)} records",expanded=True):
                for row in page_rows[-50:]: st.markdown(f'<div class="page"><div><b>Page {ui._int(row.get("page_number"))}</b><small>{esc(row.get("updated_at"))}</small></div><div>{status(row.get("extraction_status"))}</div><div>OCR · {esc(row.get("ocr_status") or "not required")}</div><div>{len(str(row.get("text") or "")):,} chars</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)


def ask_page(system) -> None:
    topbar(system,"Ask BookRAG"); available=ui.ready_docs(system); names=["All ready documents"]+[ui._doc_name(d) for d in available]
    st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Grounded research console</div><div class="sectionsub">Question first. Evidence and diagnostics appear after a result exists.</div></div><span class="chip">READY SOURCES · {len(available)}</span></div>',unsafe_allow_html=True)
    scope=st.selectbox("Source scope",names,key="renovation_ask_scope"); selected=None if scope==names[0] else {"document_id":available[names.index(scope)-1].get("document_id")}
    st.caption("Quick questions")
    for preset in ["What are the main findings?","What limitations are reported?","Compare the most relevant results."]:
        if st.button(preset,key="renovation_preset_"+hashlib.sha1(preset.encode()).hexdigest()[:8],use_container_width=True): st.session_state["research_question"]=preset
    question=st.text_area("Research question",height=132,placeholder="Ask about a population, intervention, outcome, mechanism, limitation, or comparison…",key="research_question")
    a,b,c=st.columns([2,1,1])
    with a: run=st.button("Run grounded search",type="primary",use_container_width=True,key="renovation_ask_run")
    with b:
        if st.button("Regenerate",use_container_width=True,disabled=not bool(st.session_state.get("answer_result")),key="renovation_regenerate"): run=True
    with c:
        if st.button("Clear",use_container_width=True,key="renovation_clear_chat"):
            st.session_state.pop("answer_result",None); st.session_state["chat_history"]=[]; st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)
    if run:
        if not question.strip(): st.warning("Enter a research question first.")
        elif not available: st.warning("No READY document is available yet.")
        else:
            with st.spinner("Retrieving evidence and composing a grounded response…"):
                try:
                    result=system.answer(question.strip(),metadata_filter=selected); st.session_state["answer_result"]=result
                    history=st.session_state.setdefault("chat_history",[]); history.append({"role":"user","content":question.strip()}); history.append({"role":"assistant","content":str(result.get("answer") or "")})
                except Exception as exc: st.error(f"The answer could not be produced safely: {exc}")
    result=st.session_state.get("answer_result")
    if not isinstance(result,dict):
        with st.expander("How to ask stronger questions",expanded=True): st.write("Ask about one measurable concept, population, intervention, outcome, mechanism, or comparison. BookRAG will abstain when evidence is insufficient.")
        return
    answer=str(result.get("answer") or ""); evidence=ui._evidence(result); grounded=not bool(result.get("abstained")) and str(result.get("status","")).upper() not in {"ABSTAIN","SYSTEM_NOT_READY"}
    left,right=st.columns([1.28,.72])
    with left:
        st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Answer</div><div class="sectionsub">{("Grounded response" if grounded else "Abstention / caution")} · {len(evidence)} evidence item(s)</div></div>{status("GROUNDED" if grounded else "ABSTAIN")}</div><div class="answertext">{esc(answer)}</div></div>',unsafe_allow_html=True)
        values=[("Answerability",ui._metric(result,"answerability"),"evidence alignment"),("Evidence confidence",ui._metric(result,"evidence_confidence","confidence"),"retrieval support"),("Query quality",ui._metric(result,"query_quality"),"question signal"),("Decision",ui._metric(result,"decision") or ("ABSTAIN" if result.get("abstained") else "GROUNDED"),"grounding gate")]
        metric_html=[]
        for label,value,hint in values:
            display=f"{ui._float(value)*100:.0f}%" if isinstance(value,(int,float)) and label not in {"Query quality","Decision"} else str(value if value is not None else "—")
            metric_html.append(f'<div class="metric"><label>{esc(label)}</label><b>{esc(display)}</b><small>{esc(hint)}</small></div>')
        st.markdown('<div class="metricgrid">'+''.join(metric_html)+'</div>',unsafe_allow_html=True)
        st.markdown('<div class="section"><div class="sectiontitle">Evidence</div><div class="sectionsub">The passages returned with the answer.</div>',unsafe_allow_html=True)
        if evidence:
            for i,item in enumerate(evidence,1):
                title,page,score,snippet=ui._evidence_row(item,i); score_value=ui._float(score,default=-1); score_text=f"{score_value:.3f}" if score_value>=0 else str(score)
                st.markdown(f'<div class="evidence"><div class="evidencehead"><div class="docname">{esc(title)}</div><div class="evidencemeta">Page {esc(page)} · score {esc(score_text)}</div></div><div class="snippet">{esc(snippet[:1600])}</div></div>',unsafe_allow_html=True)
        else: st.markdown('<div class="empty">No evidence records were returned with this answer.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><div class="sectiontitle">Answer tools</div><div class="sectionsub">Export the answer or open diagnostics without making them compete with the result.</div>',unsafe_allow_html=True)
        st.download_button("Export Markdown",data=answer.encode(),file_name="bookrag_answer.md",mime="text/markdown",use_container_width=True)
        st.download_button("Export JSON",data=json.dumps(result,default=str,indent=2).encode(),file_name="bookrag_answer.json",mime="application/json",use_container_width=True)
        with st.expander("Query trace",expanded=False): st.json(result.get("query_trace",{}))
        with st.expander("Raw response",expanded=False): st.json(result)
        st.markdown('</div>',unsafe_allow_html=True)


def inspector_page(system) -> None:
    topbar(system,"Inspector"); documents=ui.docs(system)
    if not documents: st.markdown('<div class="section"><div class="empty">No document is available to inspect.</div></div>',unsafe_allow_html=True); return
    ids=[str(d.get("document_id")) for d in documents]; selected_id=str(st.session_state.get("inspect_doc_id") or ""); default=ids.index(selected_id) if selected_id in ids else 0
    index=st.selectbox("Document",range(len(documents)),index=default,format_func=lambda i:ui._doc_name(documents[i]),key="renovation_inspect_selector"); document=documents[index]; doc_id=str(document.get("document_id") or ""); st.session_state["inspect_doc_id"]=doc_id
    stage,description=ui._stage(document); progress=ui._progress(document)
    st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">{esc(ui._doc_name(document))}</div><div class="sectionsub">{esc(description)}</div></div>{status(document.get("status"))}</div><div class="meter"><i style="width:{progress*100:.1f}%"></i></div>',unsafe_allow_html=True)
    vals=[("Stage",stage),("Pages",f'{ui._int(document.get("current_page"))} / {ui._int(document.get("total_pages"))}'),("Chunks",f'{ui._int(document.get("chunk_count")):,}'),("Embeddings",f'{ui._int(document.get("embedding_count")):,}'),("Embedding dimension",document.get("embedding_dimension") or "—"),("Parser",document.get("parser_version") or "—")]
    st.markdown('<div class="metricgrid">'+''.join(f'<div class="metric"><label>{esc(k)}</label><b>{esc(v)}</b></div>' for k,v in vals)+'</div>',unsafe_allow_html=True)
    if document.get("error"): st.error(str(document.get("error")))
    if st.button("Verify index",type="primary",use_container_width=True,key="renovation_verify_index"):
        try: st.session_state["verify_result"]=system.verify_index(doc_id)
        except Exception as exc: st.error(str(exc))
    if st.session_state.get("verify_result") is not None:
        with st.expander("Verification result",expanded=True): st.json(st.session_state.pop("verify_result"))
    st.markdown('</div>',unsafe_allow_html=True)
    rows=ui.pages(system,doc_id)
    st.markdown('<div class="section"><div class="sectiontitle">Page records</div><div class="sectionsub">Durable extraction and OCR checkpoints for every physical page.</div>',unsafe_allow_html=True)
    if rows:
        for row in rows: st.markdown(f'<div class="page"><div><b>Page {ui._int(row.get("page_number"))}</b><small>{esc(row.get("updated_at"))}</small></div><div>{status(row.get("extraction_status"))}</div><div>OCR · {esc(row.get("ocr_status") or "not required")}</div><div>{len(str(row.get("text") or "")):,} chars</div></div>',unsafe_allow_html=True)
    else: st.markdown('<div class="empty">No page records stored yet.</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
    with st.expander("Event timeline",expanded=True):
        events=ui.events(system,doc_id,250)
        if not events: st.caption("No events recorded for the current ingestion attempt.")
        else:
            st.markdown('<div class="timeline">',unsafe_allow_html=True)
            for event in reversed(events[-50:]):
                message=event.get("message") or event.get("event_type") or event.get("stage") or "Event"
                st.markdown(f'<div class="event"><b>{esc(message)}</b> · {status(event.get("status") or event.get("stage"))}<small>{esc(event.get("created_at"))} · page {esc(event.get("current_page") or "—")} / {esc(event.get("total_pages") or "—")}</small></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
    with st.expander("Raw state",expanded=False): st.json(document)


@st.fragment(run_every="8s")
def system_snapshot(system) -> None:
    try: report=system.health_report()
    except Exception as exc: report={"ready":False,"error":str(exc)}
    try: ollama_ok,ollama_message,models=ui.ollama_health(system.settings.ollama_base_url)
    except Exception as exc: ollama_ok,ollama_message,models=False,str(exc),[]
    embedding=report.get("embedding",{}) if isinstance(report,dict) else {}; index=report.get("index",{}) if isinstance(report,dict) else {}; contract=report.get("feature_contract",{}) if isinstance(report,dict) else {}
    rows=[("Ollama","ONLINE" if ollama_ok else "OFFLINE",ollama_message),("Embedding","PASS" if embedding.get("ok") else "UNKNOWN",embedding.get("identity") or embedding.get("error") or system.settings.embedding_model),("Vector index",str(index.get("status","UNKNOWN")).upper(),index.get("error") or "Runtime-reported index health"),("Production contract","PASS" if contract.get("all_resolved",False) else "CHECK","Feature resolution")]
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">System health</div><div class="sectionsub">Operational signals refresh automatically.</div></div><span class="chip">AUTO</span></div>',unsafe_allow_html=True)
    for label,state,detail in rows: st.markdown(f'<div class="page"><div><b>{esc(label)}</b></div><div>{status(state)}</div><div>{esc(detail)}</div><div></div></div>',unsafe_allow_html=True)
    if models: st.caption("Ollama models · "+", ".join(models))
    if report.get("error"): st.error(str(report["error"]))
    st.markdown('</div>',unsafe_allow_html=True)


def system_page(system) -> None:
    topbar(system,"System"); system_snapshot(system); left,right=st.columns(2)
    with left:
        st.markdown('<div class="section"><div class="sectiontitle">Runtime profile</div><div class="sectionsub">Configured local services and index scale.</div>',unsafe_allow_html=True)
        for key,value in [("Ollama",system.settings.ollama_base_url),("Embedding",system.settings.embedding_model),("Generation",system.settings.generation_model),("Top K",system.settings.top_k),("Search sections",ui.total_chunks(system)),("Embeddings",ui.total_embeddings(system))]: st.markdown(f'<div class="metric" style="margin-top:8px"><label>{esc(key)}</label><b>{esc(value)}</b></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section"><div class="sectiontitle">Operational meanings</div><div class="sectionsub">How to interpret the states shown across the product.</div>',unsafe_allow_html=True)
        for key,value in [("READY","Document is published to retrieval."),("PROCESSING","Persistent page state is still changing."),("FAILED","Inspect the document error and timeline."),("ABSTAIN","Evidence did not justify a confident answer."),("OFFLINE","The local model endpoint is unavailable; generation cannot safely proceed.")]: st.markdown(f'<div class="evidence"><div class="evidencetitle">{esc(key)}</div><div class="snippet">{esc(value)}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)


def settings_page(system) -> None:
    topbar(system,"Settings"); settings=system.settings
    st.markdown('<div class="section"><div class="sectiontitle">Research behavior</div><div class="sectionsub">Safe runtime controls remain validated by the application layer.</div>',unsafe_allow_html=True)
    with st.form("renovation_settings_form"):
        host=st.text_input("Ollama host",str(settings.ollama_base_url)); embedding_model=st.text_input("Embedding model",str(settings.embedding_model)); generation_model=st.text_input("Generation model",str(settings.generation_model))
        a,b=st.columns(2)
        with a: top_k=st.number_input("Retrieval results",1,50,int(settings.top_k)); vector_weight=st.slider("Semantic weight",0.0,1.0,float(settings.vector_weight),0.05)
        with b: temperature=st.slider("Generation temperature",0.0,1.0,float(settings.temperature),0.05); neighbor=st.checkbox("Neighbor expansion",bool(settings.neighbor_expansion))
        save=st.form_submit_button("Save configuration",type="primary",use_container_width=True)
    if save:
        try:
            ok,warnings=system.apply_settings_in_place({"ollama_base_url":host,"embedding_model":embedding_model,"generation_model":generation_model,"top_k":int(top_k),"vector_weight":float(vector_weight),"temperature":float(temperature),"neighbor_expansion":bool(neighbor)})
            if ok: st.success("Configuration saved.")
            for warning in warnings or []: st.warning(warning)
        except Exception as exc: st.error(f"Settings were not saved: {exc}")
    st.markdown('</div>',unsafe_allow_html=True)
    with st.expander("Advanced diagnostics",expanded=False): st.json({"project_root":str(settings.project_root),"incoming_dir":str(settings.incoming_dir),"vector_db_dir":str(settings.vector_db_dir),"ingestion_db_path":str(settings.ingestion_db_path)})


def apply() -> None:
    ui.css=inject_css
    ui.sidebar=sidebar
    ui.topbar=topbar
    ui.home=home
    ui.documents_page=documents_page
    ui.processing_live=processing_page
    ui.processing_page=processing_page
    ui.ask_page=ask_page
    ui.inspector_page=inspector_page
    ui.system_snapshot=system_snapshot
    ui.system_page=system_page
    ui.settings_page=settings_page
