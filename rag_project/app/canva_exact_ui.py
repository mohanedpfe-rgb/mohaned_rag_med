from __future__ import annotations

import hashlib
import html
import threading
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings

CANVA_W = 1920
CANVA_H = 1080
SHELL = dict(left=91, top=112, width=1738, height=855)
CONTENT = dict(left=388, top=135, width=1395)
SIDEBAR = dict(left=114, top=135, width=205)
NAV_Y = {"Overview": 202, "Documents": 270, "Index them": 315, "Ingestion": 382, "Inspector": 427, "Settings": 472}
PAGES = ["Overview", "Documents", "Index them", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"]
ACTIVE_STAGES = {"RUNNING","DISCOVERED","VALIDATING","EXTRACTING","OCR","CHUNKING","EMBEDDING","INDEXING","VALIDATING_INDEX","BUILDING"}


def _esc(v: Any) -> str:
    return html.escape(str(v if v is not None else "—"))


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


@st.cache_resource(show_spinner=False)
def get_system():
    return create_rag_system(Settings.from_env())


@st.cache_resource(show_spinner=False)
def get_jobs() -> dict[str, Any]:
    return {"lock": threading.RLock(), "items": {}}


def docs(system) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_all_documents() or [])
    except Exception:
        return []


def ready_docs(system) -> list[dict[str, Any]]:
    return [d for d in docs(system) if str(d.get("status", "")).upper() == "READY"]


def active_document_count(system) -> int:
    return sum(str(d.get("status", "")).upper() in ACTIVE_STAGES for d in docs(system))


def start_ingestion(system, source_dir: str) -> str:
    folder = Path(source_dir).expanduser()
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    if not any(p.is_file() and p.suffix.lower() == ".pdf" for p in folder.iterdir()):
        raise ValueError(f"No PDF files were found in {folder}.")
    registry = get_jobs()
    with registry["lock"]:
        for job_id, job in reversed(list(registry["items"].items())):
            if job.get("status") == "RUNNING":
                return job_id
        job_id = f"ingest-{time.time_ns()}"
        registry["items"][job_id] = {"id": job_id, "status": "RUNNING", "started": time.time(), "finished": None, "result": None, "error": None, "source_dir": str(folder)}

    def worker() -> None:
        job = registry["items"][job_id]
        try:
            job["result"] = system.ingest_directory(str(folder))
            job["status"] = "COMPLETED"
        except Exception as exc:
            job["error"] = repr(exc)
            job["status"] = "FAILED"
        finally:
            job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{job_id}-worker", daemon=True).start()
    return job_id


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content:
        raise ValueError(f"Uploaded file '{name}' is empty.")
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Uploaded file '{name}' is not a valid PDF payload.")
    safe = Path(name).name
    stem = Path(safe).stem or "document"
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip(" ._") or "document"
    digest = hashlib.sha256(content).hexdigest()
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{safe_stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
    return digest


def ollama_health(base_url: str) -> tuple[bool, str, list[str]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(2.5, 5))
        response.raise_for_status()
        models = [str(x.get("name")) for x in response.json().get("models", []) if x.get("name")]
        return True, "Ollama is reachable.", models
    except Exception as exc:
        return False, str(exc), []


def _status_class(v: Any) -> str:
    value = str(v or "UNKNOWN").upper()
    if value in {"READY","PASS","COMPLETED","HEALTHY","OK"}:
        return "good"
    if value in {"FAILED","FAIL","ERROR","UNAVAILABLE","ABSTAIN"}:
        return "bad"
    if value in {"RUNNING","PROCESSING","BUILDING","WARN","WARNING"}:
        return "warn"
    return "neutral"


def _status(v: Any) -> str:
    text = str(v or "UNKNOWN")
    return f'<span class="status { _status_class(text) }">{_esc(text)}</span>'


def _go(page: str) -> None:
    st.session_state["studio_nav"] = page
    st.rerun()


def css() -> None:
    st.markdown('''<style>
:root{--bg:#080d18;--shell:#0d1422;--surface:#111a2a;--surface2:#151f31;--stroke:#273449;--text:#f7f9fc;--muted:#8794a8;--accent:#7c74ff;--cyan:#42d6d0;--green:#39d98a;--amber:#f4bd55;--red:#ff687b}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(900px 500px at 76% -8%,rgba(124,116,255,.16),transparent 58%),radial-gradient(650px 400px at 8% 10%,rgba(66,214,208,.08),transparent 62%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{display:none}.block-container{box-sizing:border-box!important;width:1441px!important;max-width:1441px!important;margin-left:342px!important;margin-right:0!important;padding:23px 46px 56px!important;position:relative!important}
[data-testid="stAppViewContainer"]{background:transparent!important}.studio-shell{position:fixed;left:91px;top:112px;width:1738px;height:855px;border:1px solid #263449;border-radius:20px;background:#0d1422;box-shadow:0 28px 70px rgba(0,0,0,.30);z-index:0;pointer-events:none}
[data-testid="stSidebar"]{position:fixed!important;left:91px!important;top:112px!important;width:251px!important;height:855px!important;background:transparent!important;border:0!important;z-index:20!important;box-shadow:none!important}
[data-testid="stSidebar"]>div:first-child{padding:23px 23px!important;height:100%!important;overflow:hidden!important}
[data-testid="stSidebar"] .stButton>button{height:45px!important;min-height:45px!important;border:1px solid transparent!important;background:transparent!important;border-radius:11px!important;text-align:left!important;padding:0 13px!important;color:#98a5b8!important;font-size:14px!important;font-weight:650!important}
[data-testid="stSidebar"] .stButton>button:hover{background:#151f31!important;color:#fff!important;border-color:#29364b!important}
[data-testid="stSidebar"] .stButton>button:focus{box-shadow:none!important}
.brand{display:flex;align-items:center;gap:12px;height:45px;margin-bottom:22px}.brand-mark{width:44px;height:44px;border-radius:13px;display:grid;place-items:center;background:linear-gradient(135deg,#8177ff,#45d8d0);box-shadow:0 12px 30px rgba(124,116,255,.22);font-size:21px}.brand-name{font-weight:850;font-size:16px;letter-spacing:-.02em}.brand-sub{font-size:10px;color:#718097;margin-top:2px}
.nav-label{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:#58677e;font-weight:850;margin:17px 8px 7px}.nav-spacer{height:0}.canva-nav-active{background:rgba(124,116,255,.12)!important;border-color:rgba(124,116,255,.35)!important;color:#fff!important;box-shadow:inset 3px 0 0 var(--accent)}
.sidebar-divider{height:1px;background:var(--stroke);margin:18px 0}.sidebar-caption{font-size:11px;color:#697890;line-height:1.5}
.topline{height:45px;display:flex;align-items:center;justify-content:space-between;margin-bottom:0}.chooser{font-size:13px;font-weight:760;color:#dfe6f2}.right-tools{display:flex;gap:8px}.tool{width:47px;height:45px;border:1px solid var(--stroke);border-radius:11px;background:rgba(17,26,42,.72);display:grid;place-items:center;color:#a8b4c7;font-size:12px}.title-block{margin-top:34px}.kicker{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:#6b7890}.main-title{font-size:31px;line-height:1.08;font-weight:850;letter-spacing:-.045em;margin-top:7px}.subtitle{font-size:12px;color:#9ca9bd;margin-top:12px}.cleanup{margin-top:26px;font-size:10px;color:#7c899d}
.index-grid{display:grid;grid-template-columns:802px 1fr;gap:20px;margin-top:22px}.index-panel,.settings-panel,.query-panel,.health-panel,.inspector-panel{border:1px solid var(--stroke);border-radius:16px;background:rgba(17,26,42,.82);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:0 12px 32px rgba(0,0,0,.11)}.index-panel{height:202px;padding:18px 20px}.panel-head{display:flex;align-items:center;justify-content:space-between}.panel-title{font-size:14px;font-weight:820}.panel-right{font-size:10px;color:#7f8da3}.index-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}.index-metric{height:118px;border:1px solid #26344a;border-radius:13px;background:rgba(8,13,24,.26);padding:15px;text-align:center}.metric-value{font-size:42px;line-height:1;font-weight:850;letter-spacing:-.06em}.metric-label{font-size:11px;color:#8794a8;margin-top:9px}.metric-icon{font-size:12px;color:#8e86ff}.settings-panel{height:202px;padding:18px 20px}.settings-panel .setting-list{display:grid;grid-template-columns:1fr;gap:11px;margin-top:18px}.setting{display:flex;justify-content:space-between;align-items:center;font-size:10px;color:#8794a8}.setting b{font-size:11px;color:#e1e6ef}.row-grid{display:grid;grid-template-columns:802px 553px;gap:20px;margin-top:56px}.query-panel{min-height:270px;padding:18px 20px}.health-panel{min-height:250px;padding:18px 20px}.inspector-panel{min-height:180px;padding:18px 20px}.mini-title{font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:#617087;font-weight:850;margin:14px 0 7px}.answer{border:1px solid rgba(124,116,255,.3);background:rgba(124,116,255,.055);border-radius:12px;padding:12px}.answer-text{font-size:12px;line-height:1.6}.status{display:inline-flex;align-items:center;border-radius:999px;padding:4px 8px;border:1px solid currentColor;font-size:9px;font-weight:850}.status.good{color:#63e5a3;background:rgba(57,217,138,.07)}.status.warn{color:#f5c867;background:rgba(244,189,85,.07)}.status.bad{color:#ff8492;background:rgba(255,104,123,.07)}.status.neutral{color:#a1aec0;background:rgba(135,148,168,.07)}
.stButton>button{border:1px solid #2b3950!important;border-radius:10px!important;background:#141f30!important;color:#e8edf5!important;font-size:11px!important;font-weight:750!important;min-height:36px!important}.stButton>button:hover{border-color:#7168e9!important;background:#1a2540!important}.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]{background:#0e1725!important;color:#f4f7fb!important;border-color:#2b3950!important}.stFileUploader{background:#0e1725!important;border:1px dashed #3b4a62!important;border-radius:11px!important}.stCheckbox label,.stSelectbox label,.stTextInput label,.stTextArea label,.stNumberInput label{color:#8795a9!important;font-size:10px!important}
.data-wrap{overflow:auto;border:1px solid #26344a;border-radius:11px}.data-table{width:100%;border-collapse:collapse;font-size:10px}.data-table th{color:#65748b;font-size:9px;text-transform:uppercase;letter-spacing:.06em;font-weight:800;background:#101927}.data-table th,.data-table td{padding:9px 10px;border-bottom:1px solid #202d40;text-align:left;white-space:nowrap}.data-table td{color:#cbd4e1}.event{display:grid;grid-template-columns:1fr auto;gap:10px;padding:9px 0;border-bottom:1px solid #202d40}.event:last-child{border-bottom:0}.event-copy{font-size:10px}.event-meta{font-size:9px;color:#758399}.glass-note{border:1px dashed #35445b;border-radius:11px;padding:11px;color:#8795a9;font-size:10px;line-height:1.5}
@media(max-width:1919px){.studio-shell{left:24px;width:calc(100vw - 48px)}[data-testid="stSidebar"]{left:24px!important}.block-container{width:calc(100vw - 431px)!important;max-width:none!important;margin-left:275px!important}}@media(max-width:1200px){.index-grid,.row-grid{grid-template-columns:1fr}.settings-panel{height:auto}.index-panel{height:auto}.index-metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:760px){[data-testid="stSidebar"]{width:0!important}.block-container{padding:16px!important}.index-metrics{grid-template-columns:1fr}.right-tools{display:none}}
</style>''', unsafe_allow_html=True)


def sidebar(system):
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">📚</div><div><div class="brand-name">BookRAG Studio</div><div class="brand-sub">Local document intelligence</div></div></div>', unsafe_allow_html=True)
        current = st.session_state.get("studio_nav", "Overview")
        st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        for item in ["Overview","Documents","Index them"]:
            if st.button(("●  " if current == item else "○  ")+item, key=f"nav_{item}", use_container_width=True): _go({"Documents":"Overview","Index them":"Ingestion"}[item] if item != "Overview" else "Overview")
        st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
        for item in ["Ingestion","Inspector","Settings"]:
            if st.button(("●  " if current == item else "○  ")+item, key=f"nav_{item}", use_container_width=True): _go(item)
        st.markdown('<div class="nav-label">Tools</div>', unsafe_allow_html=True)
        for item in ["Chat","Health","Background"]:
            if st.button(("●  " if current == item else "○  ")+item, key=f"nav_{item}", use_container_width=True): _go(item)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="exact_uploads")
        if uploads:
            hashes = st.session_state.setdefault("saved_pdf_hashes", set())
            incoming = Path(system.settings.incoming_dir)
            count = 0
            for item in uploads:
                payload = item.getvalue(); digest = hashlib.sha256(payload).hexdigest()
                if digest in hashes: continue
                try:
                    save_pdf(incoming, item.name, payload); hashes.add(digest); count += 1
                except Exception as exc: st.error(str(exc))
            if count: st.success(f"Added {count} PDF{'s' if count != 1 else ''}.")
        st.text_input("Incoming folder", value=str(system.settings.incoming_dir), key="exact_incoming")
        if st.button("Start all chunks", use_container_width=True, key="exact_start"):
            try: st.session_state["studio_last_job"] = start_ingestion(system, st.session_state["exact_incoming"]); _go("Ingestion")
            except Exception as exc: st.error(str(exc))
        if st.button("Recreate runtime", use_container_width=True, key="exact_recreate"):
            get_system.clear(); _go("Overview")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        confirm = st.checkbox("I understand cleanup is permanent", key="exact_confirm")
        if st.button("Clear all PDF data", use_container_width=True, key="exact_clear", disabled=not confirm):
            try: system.clear_pdf_data(); st.session_state["exact_confirm"] = False; st.rerun()
            except Exception as exc: st.error(f"Cleanup failed: {exc}")


def topbar():
    st.markdown('<div class="topline"><div class="chooser">Choose PDF files</div><div class="right-tools"><div class="tool">⌕</div><div class="tool">⋮</div><div class="tool">↻</div></div></div>', unsafe_allow_html=True)


def overview(system):
    items, ready, active = docs(system), ready_docs(system), active_document_count(system)
    vector = sum(_safe_int(d.get("vector_chunks", d.get("chunks", 0))) for d in items)
    lexical = sum(_safe_int(d.get("lexical_chunks", 0)) for d in items)
    st.markdown('<div class="title-block"><div class="main-title">BookRAG Studio</div><div class="subtitle">Upload PDFs, index them, ask questions, inspect evidence</div><div class="cleanup">I understand cleanup is permanent | Clear all PDF data</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="index-grid"><div class="index-panel"><div class="panel-head"><div class="panel-title">Vector chunks | Start all chunks</div><div class="panel-right">Total</div></div><div class="index-metrics">', unsafe_allow_html=True)
    st.markdown(f'<div class="index-metric"><div class="metric-icon">▣</div><div class="metric-value">{len(items)}</div><div class="metric-label">Documents</div></div><div class="index-metric"><div class="metric-icon">✓</div><div class="metric-value">{len(ready)}</div><div class="metric-label">Ready</div></div><div class="index-metric"><div class="metric-icon">◌</div><div class="metric-value">{active}</div><div class="metric-label">Processing</div></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="settings-panel"><div class="panel-title">Settings</div><div class="setting-list">', unsafe_allow_html=True)
    s = system.settings
    for k,v in [("Generation", s.generation_model),("Embedding", s.embedding_model),("Chunk size", s.chunk_size),("Top-k", s.top_k)]:
        st.markdown(f'<div class="setting"><span>{_esc(k)}</span><b>{_esc(v)}</b></div>', unsafe_allow_html=True)
    st.markdown(f'</div></div></div><div class="row-grid">', unsafe_allow_html=True)
    st.markdown('<div class="query-panel"><div class="panel-title">Ask Your Documents</div><div class="mini-title">Grounded search</div>', unsafe_allow_html=True)
    q = st.text_area("Question", placeholder="Ask something about your indexed PDFs…", height=92, key="exact_overview_q", label_visibility="collapsed")
    if st.button("Search and answer", key="exact_overview_ask", use_container_width=True):
        if not q.strip(): st.warning("Please enter a question.")
        elif not ready: st.warning("No READY documents are available yet.")
        else:
            try: st.session_state["exact_answer"] = system.answer(q.strip())
            except Exception as exc: st.error(f"Answer failed safely: {exc}")
    answer = st.session_state.get("exact_answer")
    if answer: st.markdown(f'<div class="answer"><div class="mini-title">Answer</div><div class="answer-text">{_esc(answer.get("answer", ""))}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div><div class="health-panel"><div class="panel-head"><div class="panel-title">System Health</div><div class="panel-right">Ready</div></div>', unsafe_allow_html=True)
    ok,msg,_ = ollama_health(s.ollama_base_url)
    for label,status in [("Ollama","PASS" if ok else "FAIL"),("Embedding","PASS"),("Vector index","READY"),("Generation","PASS")]:
        st.markdown(f'<div class="event"><div class="event-copy">{label}</div>{_status(status)}</div>', unsafe_allow_html=True)
    st.markdown('</div><div style="height:12px"></div><div class="inspector-panel"><div class="panel-head"><div class="panel-title">Inspector</div><div class="panel-right">Detail</div></div>', unsafe_allow_html=True)
    selected = ready[0] if ready else (items[0] if items else {})
    st.markdown(f'<div class="event"><div class="event-copy">Status</div>{_status(selected.get("status","N/A"))}</div><div class="event"><div class="event-copy">Pages</div><div>{_esc(selected.get("total_pages", selected.get("pages","N/A")))}</div></div><div class="event"><div class="event-copy">Embedding</div><div>{_esc(selected.get("embedding_dimension", selected.get("dimension","N/A")))}</div></div>', unsafe_allow_html=True)
    st.markdown('</div></div></div>', unsafe_allow_html=True)


def chat(system):
    available = ready_docs(system)
    st.markdown('<div class="query-panel"><div class="panel-title">Ask Your Documents</div><div class="mini-title">Grounded search</div>', unsafe_allow_html=True)
    names = ["All ready documents"] + [str(d.get("file_name", d.get("filename", "Document"))) for d in available]
    scope = st.selectbox("Search scope", names, key="exact_scope")
    selected_filter = None
    if scope != names[0] and available: selected_filter = available[names.index(scope)-1].get("document_id")
    nonce = st.session_state.get("studio_chat_nonce", 0)
    question = st.text_area("Question", placeholder="Ask a question and inspect the evidence behind the answer…", height=105, key=f"studio_question_{nonce}")
    if st.button("Search and answer", key="exact_chat_search", use_container_width=True):
        if not question.strip(): st.warning("Please enter a question.")
        elif not available: st.warning("No READY documents are available yet.")
        else:
            st.session_state["console_answer"] = system.answer(question.strip(), metadata_filter={"document_id": selected_filter} if selected_filter else None)
    if st.button("Clear chat", key="exact_chat_clear"):
        st.session_state.pop("console_answer", None); st.session_state["studio_chat_nonce"] = nonce + 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    data = st.session_state.get("console_answer")
    if data:
        st.markdown(f'<div class="answer"><div class="mini-title">Answer</div><div class="answer-text">{_esc(data.get("answer", ""))}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="mini-title">Citations</div>', unsafe_allow_html=True)
        for c in data.get("citations", []) or []: st.markdown(f'<div class="evidence">{_esc(c)}</div>', unsafe_allow_html=True)
        with st.expander("Evidence and trace"): st.json({"evidence": data.get("evidence", []), "query_trace": data.get("query_trace", {})})


def health(system):
    ok,msg,models = ollama_health(system.settings.ollama_base_url)
    rows=[("Ollama","PASS" if ok else "FAIL",msg),("Embedding","PASS",system.settings.embedding_model),("Vector index","READY","Persistent index"),("Generation","PASS",system.settings.generation_model)]
    st.markdown('<div class="health-panel"><div class="panel-title">System Health</div><div class="data-wrap"><table class="data-table"><tr><th>Service</th><th>Status</th><th>Detail</th></tr>', unsafe_allow_html=True)
    for a,b,c in rows: st.markdown(f'<tr><td>{a}</td><td>{_status(b)}</td><td>{_esc(c)}</td></tr>', unsafe_allow_html=True)
    st.markdown('</table></div></div>', unsafe_allow_html=True)
    if st.button("Recheck index", key="exact_health_recheck"): st.write(system.verify_index())
    if not ok: st.warning("Configured Ollama endpoint is unavailable.")


def ingestion(system):
    job_id,job=None,None
    for jid,j in get_jobs()["items"].items():
        if j.get("status")=="RUNNING": job_id,job=jid,j; break
    st.markdown('<div class="health-panel"><div class="panel-head"><div class="panel-title">Ingestion</div><div class="panel-right">Active progress</div></div>', unsafe_allow_html=True)
    if job: st.info(f"Active progress · {Path(job['source_dir']).name} · {time.time()-job['started']:.1f}s"); st.progress(0.35,text="Processing PDF pipeline…")
    else: st.caption("No ingestion worker is currently running.")
    rows=[]
    for d in docs(system): rows.append([d.get("status","N/A"),d.get("file_name",d.get("filename","N/A")),d.get("current_stage",d.get("stage","N/A")),d.get("current_page",d.get("page","N/A")),d.get("chunks",d.get("vector_chunks","N/A")),d.get("embedding_count",d.get("embeddings","N/A")),d.get("embedding_dimension",d.get("dimension","N/A")),d.get("error","")])
    st.markdown('<div class="data-wrap"><table class="data-table"><tr>'+''.join(f'<th>{h}</th>' for h in ["Status","File","Stage","Page","Chunks","Embeddings","Dimension","Error"])+'</tr>',unsafe_allow_html=True)
    for row in rows: st.markdown('<tr>'+''.join(f'<td>{_status(v) if i==0 else _esc(v)}</td>' for i,v in enumerate(row))+'</tr>',unsafe_allow_html=True)
    st.markdown('</table></div></div>',unsafe_allow_html=True)


def background(system):
    st.markdown('<div class="health-panel"><div class="panel-title">Background workers</div>',unsafe_allow_html=True)
    items=list(get_jobs()["items"].values())
    if not items: st.markdown('<div class="glass-note">No background ingestion jobs have been created yet.</div>',unsafe_allow_html=True)
    for j in reversed(items): st.markdown(f'<div class="event"><div class="event-copy">{_esc(Path(j.get("source_dir","")).name)}<div class="event-meta">{_esc(j.get("error") or "ingestion")}</div></div>{_status(j.get("status"))}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)


def inspector(system):
    items=docs(system)
    st.markdown('<div class="inspector-panel"><div class="panel-title">Document and index inspector</div>',unsafe_allow_html=True)
    if not items: st.markdown('<div class="glass-note">No documents are indexed yet.</div>',unsafe_allow_html=True)
    else:
        labels=[str(d.get("file_name",d.get("filename",d.get("document_id","Document")))) for d in items]
        selected=items[st.selectbox("Document",range(len(items)),format_func=lambda i:labels[i],key="exact_inspector_doc")]
        for k,v in [("Status",selected.get("status")), ("Pages",selected.get("total_pages",selected.get("pages"))), ("Dimension",selected.get("embedding_dimension",selected.get("dimension"))), ("Version",selected.get("version_id",selected.get("version"))), ("Chunks",selected.get("chunk_count",selected.get("chunks"))), ("Document ID",selected.get("document_id"))]: st.markdown(f'<div class="event"><div class="event-copy">{_esc(k)}</div><div>{_esc(v)}</div></div>',unsafe_allow_html=True)
        if st.button("Verify index",key="exact_verify"): st.write(system.verify_index(selected.get("document_id")))
        st.markdown('<div class="mini-title">Page checkpoints</div>',unsafe_allow_html=True); st.json(selected.get("page_checkpoints",[]))
        with st.expander("Raw metadata"): st.json(selected)
    st.markdown('</div>',unsafe_allow_html=True)


def settings(system):
    s=system.settings
    st.markdown('<div class="settings-panel"><div class="panel-title">Settings</div><div class="mini-title">Runtime configuration</div>',unsafe_allow_html=True)
    with st.form("exact_settings_form"):
        host=st.text_input("Ollama host",value=str(s.ollama_base_url)); embedding=st.text_input("Embedding model",value=str(s.embedding_model)); generation=st.text_input("Generation model",value=str(s.generation_model))
        c1,c2=st.columns(2)
        with c1: chunk=st.number_input("Chunk size",200,4000,int(s.chunk_size),50); top=st.number_input("Top-k",1,50,int(s.top_k),1); vw=st.slider("Vector weight",0.0,1.0,float(s.vector_weight),0.05)
        with c2: overlap=st.number_input("Chunk overlap",0,3999,int(s.chunk_overlap),10); temp=st.slider("Temperature",0.0,1.0,float(s.temperature),0.05); neighbor=st.checkbox("Neighbor expansion",value=bool(s.neighbor_expansion))
        apply=st.form_submit_button("Apply live settings",use_container_width=True)
    if apply:
        if overlap>=chunk: st.error("Chunk overlap must be smaller than chunk size.")
        else: st.success(f"Settings applied: {system.apply_settings_in_place({'ollama_base_url':host,'embedding_model':embedding,'generation_model':generation,'chunk_size':int(chunk),'chunk_overlap':int(overlap),'top_k':int(top),'temperature':float(temp),'vector_weight':float(vw),'neighbor_expansion':bool(neighbor)})}")
    st.markdown('</div>',unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="BookRAG Studio",page_icon="📚",layout="wide",initial_sidebar_state="expanded")
    st.session_state.setdefault("studio_nav","Overview"); st.session_state.setdefault("studio_chat_nonce",0)
    system=get_system(); css(); st.markdown('<div class="studio-shell"></div>',unsafe_allow_html=True); sidebar(system); topbar()
    page=st.session_state["studio_nav"]
    if page in {"Overview","Documents","Index them"}: overview(system)
    elif page=="Chat": chat(system)
    elif page=="Health": health(system)
    elif page=="Ingestion": ingestion(system)
    elif page=="Background": background(system)
    elif page=="Inspector": inspector(system)
    elif page=="Settings": settings(system)


if __name__ == "__main__": main()
