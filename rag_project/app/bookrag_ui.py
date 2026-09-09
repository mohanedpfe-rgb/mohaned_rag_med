from __future__ import annotations

import hashlib
import html
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from rag_project.application import create_rag_system
from rag_project.configuration.settings import Settings

PAGES = ["Home", "Documents", "Live Processing", "Ask BookRAG", "Inspector", "System", "Settings"]
ACTIVE = {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING", "INTERRUPTED", "RECOVERING"}
STAGES = {
    "RUNNING": ("Starting", "The document has entered the pipeline."),
    "DISCOVERED": ("Discovered", "The PDF was accepted and queued."),
    "VALIDATING": ("Validating", "Checking the PDF structure and safety."),
    "EXTRACTING": ("Extracting", "Reading the document page by page."),
    "OCR": ("OCR", "Recovering text from scanned pages."),
    "CHUNKING": ("Chunking", "Turning extracted content into retrieval sections."),
    "EMBEDDING": ("Embedding", "Creating semantic search vectors."),
    "INDEXING": ("Indexing", "Writing semantic and lexical search records."),
    "VALIDATING_INDEX": ("Verifying", "Checking the new index before publication."),
    "READY": ("Ready", "Available for grounded research questions."),
    "COMPLETED": ("Ready", "Available for grounded research questions."),
    "FAILED": ("Failed", "Processing stopped and needs attention."),
    "FAILED_EMBEDDING": ("Embedding failed", "Semantic indexing could not complete."),
    "INTERRUPTED": ("Interrupted", "Processing stopped before completion."),
    "RECOVERING": ("Recovering", "Preparing a safe recovery attempt."),
}


def _esc(v: Any) -> str:
    return html.escape(str(v if v is not None else "—"))


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _utc(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _elapsed(start: Any, end: Any | None = None) -> float:
    d = _utc(start)
    if not d:
        return 0.0
    e = _utc(end) or datetime.now(timezone.utc)
    return max(0.0, (e - d).total_seconds())


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


def pages(system, document_id: str) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_pages(document_id) or [])
    except Exception:
        return []


def events(system, document_id: str | None = None, limit: int = 250) -> list[dict[str, Any]]:
    try:
        return list(system.state_store.get_events(document_id, limit=limit) or [])
    except Exception:
        return []


def ready_docs(system):
    return [d for d in docs(system) if str(d.get("status", "")).upper() in {"READY", "COMPLETED"}]


def active_docs(system):
    return [d for d in docs(system) if str(d.get("status", "")).upper() in ACTIVE]


def total_chunks(system) -> int:
    return sum(_int(d.get("chunk_count", d.get("chunks", 0))) for d in docs(system))


def total_embeddings(system) -> int:
    return sum(_int(d.get("embedding_count", d.get("embeddings", 0))) for d in docs(system))


def _stage(d):
    key = str(d.get("current_stage") or d.get("status") or "RUNNING").upper()
    return STAGES.get(key, (key.replace("_", " ").title(), "Processing document."))


def _status_class(v: Any) -> str:
    x = str(v or "UNKNOWN").upper()
    if x in {"READY", "COMPLETED", "PASS", "ONLINE", "HEALTHY", "OK"}:
        return "good"
    if x in {"FAILED", "FAILED_EMBEDDING", "ERROR", "OFFLINE", "UNAVAILABLE", "ABSTAIN", "INTERRUPTED"}:
        return "bad"
    if x in ACTIVE or x in {"RUNNING", "PROCESSING", "BUILDING", "WARNING", "WARN"}:
        return "warn"
    return "neutral"


def _status(v: Any) -> str:
    x = str(v or "UNKNOWN")
    return f'<span class="pill pill-{_status_class(x)}"><i></i>{_esc(x)}</span>'


def _progress(d: dict[str, Any]) -> float:
    s = str(d.get("status") or d.get("current_stage") or "RUNNING").upper()
    base = {"RUNNING": .03, "DISCOVERED": .08, "VALIDATING": .15, "EXTRACTING": .30, "OCR": .46, "CHUNKING": .58, "EMBEDDING": .73, "INDEXING": .86, "VALIDATING_INDEX": .96, "READY": 1, "COMPLETED": 1}.get(s, 1 if s in {"FAILED", "FAILED_EMBEDDING", "INTERRUPTED"} else .03)
    total, current = _int(d.get("total_pages")), _int(d.get("current_page"))
    if s in {"EXTRACTING", "OCR"} and total:
        base += min(current / total, 1) * (.14 if s == "EXTRACTING" else .10)
    return max(0, min(base, 1))


def _navigate(page: str):
    if page in PAGES:
        st.session_state["bookrag_page"] = page
        st.rerun()


def _save_pdf(incoming: Path, name: str, content: bytes) -> str:
    if not content or not content.startswith(b"%PDF-"):
        raise ValueError(f"{name} is not a valid PDF payload.")
    incoming = Path(incoming).expanduser().resolve()
    digest = hashlib.sha256(content).hexdigest()
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in (Path(name).stem or "document")).strip(" ._") or "document"
    incoming.mkdir(parents=True, exist_ok=True)
    target = incoming / f"{stem[:80]}_{digest[:12]}.pdf"
    if not target.exists():
        target.write_bytes(content)
    return digest


def save_pdf(incoming: Path, name: str, content: bytes) -> str:
    return _save_pdf(incoming, name, content)


def start_ingestion(system, source_dir: str, *, trigger: str = "manual") -> str:
    folder = Path(source_dir).expanduser().resolve()
    root = Path(system.settings.project_root).expanduser().resolve()
    try:
        folder.relative_to(root)
    except ValueError as exc:
        raise ValueError("The incoming folder must remain inside the BookRAG project.") from exc
    pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError("No PDFs are waiting in the incoming folder.")
    jobs = get_jobs()
    with jobs["lock"]:
        for job in reversed(list(jobs["items"].values())):
            if job.get("status") == "RUNNING":
                return str(job["id"])
        jid = f"ingest-{time.time_ns()}"
        job = {"id": jid, "status": "RUNNING", "started": time.time(), "finished": None, "file_count": len(pdfs), "completed": 0, "failed": 0, "result": None, "error": None, "trigger": trigger}
        jobs["items"][jid] = job

    def worker():
        try:
            result = system.ingest_directory(str(folder)) or []
            ok = sum(1 for r in result if r.get("status") in {"success", "skipped"})
            bad = sum(1 for r in result if r.get("status") == "failed")
            with jobs["lock"]:
                job.update(result=result, completed=ok, failed=bad, status="FAILED" if bad and not ok else "COMPLETED")
        except Exception as exc:
            with jobs["lock"]:
                job.update(error=str(exc), status="FAILED")
        finally:
            with jobs["lock"]:
                job["finished"] = time.time()

    threading.Thread(target=worker, name=f"{jid}-worker", daemon=True).start()
    return jid


def auto_ingest(system, count: int):
    if not count:
        return None
    try:
        jid = start_ingestion(system, str(system.settings.incoming_dir), trigger="upload")
        st.session_state["last_ingest_job"] = jid
        return jid
    except Exception as exc:
        st.session_state["auto_ingest_error"] = str(exc)
        return None


def ollama_health(base_url: str):
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(2.5, 5))
        r.raise_for_status()
        models = [str(x.get("name")) for x in r.json().get("models", []) if x.get("name")]
        return True, "Ollama is reachable", models
    except Exception as exc:
        return False, str(exc), []


def _job_running():
    jobs = get_jobs()
    with jobs["lock"]:
        return any(j.get("status") == "RUNNING" for j in jobs["items"].values())


def css():
    st.markdown(r'''<style>
:root{--bg:#080d16;--panel:#0e1623;--panel2:#111c2b;--line:#243246;--line2:#1b293b;--text:#edf3fa;--muted:#8493a7;--faint:#56667c;--accent:#63d9d1;--blue:#7e9cff;--green:#5fe0a0;--amber:#f2c76b;--red:#ff7285;--shadow:0 24px 70px rgba(0,0,0,.25)}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.stApp{background:radial-gradient(900px 500px at 80% -10%,rgba(126,156,255,.12),transparent 60%),radial-gradient(700px 450px at 10% 0,rgba(99,217,209,.07),transparent 62%),var(--bg);color:var(--text)}[data-testid=stHeader]{height:0;background:transparent}.block-container{max-width:1500px;padding:26px 42px 70px}.stApp *{letter-spacing:-.005em}
[data-testid=stSidebar]{background:#09111d!important;border-right:1px solid var(--line)!important}[data-testid=stSidebar]>div:first-child{padding:22px 16px 30px!important}.brand{display:flex;align-items:center;gap:11px;padding:4px 7px 24px}.brandmark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#76e0d9,#718dff);display:grid;place-items:center;color:#07121b;font-size:13px;font-weight:950;box-shadow:0 10px 30px rgba(99,217,209,.16)}.brandname{font-weight:900;font-size:15px}.brandsub{font-size:9px;color:var(--faint);margin-top:2px}.navgroup{margin:17px 7px 7px;font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:#58697e;font-weight:850}
[data-testid=stSidebar] .stButton>button{min-height:40px!important;border:1px solid transparent!important;background:transparent!important;color:#8e9db0!important;border-radius:9px!important;text-align:left!important;font-size:12px!important;font-weight:700!important;padding:0 12px!important}[data-testid=stSidebar] .stButton>button:hover{background:#111d2c!important;border-color:#26364b!important;color:#edf3fa!important}.sidefoot{border-top:1px solid var(--line2);margin-top:18px;padding:14px 7px;color:#68788d;font-size:9px;line-height:1.6}
.top{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-bottom:20px}.crumb{font-size:10px;color:#62738a;text-transform:uppercase;letter-spacing:.13em;font-weight:850}.title{font-size:22px;font-weight:900;letter-spacing:-.04em;margin-top:4px}.topright{display:flex;align-items:center;gap:8px}.chip{height:32px;border:1px solid var(--line);background:rgba(14,22,35,.8);border-radius:9px;padding:0 11px;display:flex;align-items:center;font-size:10px;color:#9aabbe}.dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 11px rgba(95,224,160,.7);margin-right:7px}
.hero{border:1px solid #2a3a51;border-radius:18px;padding:26px 28px;background:linear-gradient(115deg,#0e1725,#121f32 58%,#182a38);box-shadow:var(--shadow);position:relative;overflow:hidden}.hero:after{content:"";position:absolute;right:-110px;top:-150px;width:360px;height:360px;border-radius:50%;background:radial-gradient(circle,rgba(99,217,209,.14),transparent 65%)}.kicker{font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:#76d7d1;font-weight:900}.hero h1{font-size:31px;line-height:1.08;margin:8px 0 7px;letter-spacing:-.045em}.hero p{margin:0;max-width:690px;color:#93a2b5;font-size:12px;line-height:1.65}.hero-actions{margin-top:18px}
.livebar{display:flex;align-items:center;gap:8px;margin-top:12px;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:rgba(14,22,35,.8);font-size:10px;color:#8fa0b4}.livebar .grow{flex:1}.livepulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(95,224,160,.07),0 0 12px rgba(95,224,160,.6)}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin-top:12px}.stat{border:1px solid var(--line);border-radius:13px;background:linear-gradient(180deg,rgba(17,28,43,.94),rgba(12,20,32,.94));padding:15px;min-height:94px}.statlabel{font-size:9px;color:#718198;text-transform:uppercase;letter-spacing:.08em;font-weight:800}.statvalue{font-size:27px;font-weight:900;margin-top:6px;letter-spacing:-.05em}.stathint{font-size:9px;color:#53657a;margin-top:2px}
.section{border:1px solid var(--line);border-radius:15px;background:rgba(14,22,35,.86);padding:18px;margin-top:12px;box-shadow:0 12px 36px rgba(0,0,0,.10)}.sectionhead{display:flex;justify-content:space-between;align-items:flex-start;gap:15px;margin-bottom:13px}.sectiontitle{font-size:14px;font-weight:850}.sectionsub{font-size:10px;color:#687990;line-height:1.5;margin-top:4px}.twocol{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(300px,.75fr);gap:12px}.threecol{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.docrow{border:1px solid var(--line2);border-radius:11px;padding:12px;background:#0b1421;margin-top:8px}.dochead{display:flex;justify-content:space-between;gap:12px}.docname{font-size:12px;font-weight:800;overflow-wrap:anywhere}.meta{font-size:9px;color:#6f8198;margin-top:3px}.meter{height:5px;background:#1a293a;border-radius:99px;overflow:hidden;margin:11px 0 8px}.meter>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--blue));border-radius:99px}.microgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.micro{padding:8px;border:1px solid #1c2b3e;border-radius:8px;background:#0d1725}.micro span{display:block;font-size:8px;color:#62748b;text-transform:uppercase}.micro b{display:block;font-size:10px;margin-top:3px;overflow-wrap:anywhere}.pill{display:inline-flex;align-items:center;gap:5px;border-radius:99px;border:1px solid currentColor;padding:3px 7px;font-size:8px;font-weight:850;white-space:nowrap}.pill i{width:5px;height:5px;border-radius:50%;background:currentColor}.pill-good{color:var(--green);background:rgba(95,224,160,.06)}.pill-warn{color:var(--amber);background:rgba(242,199,107,.06)}.pill-bad{color:var(--red);background:rgba(255,114,133,.06)}.pill-neutral{color:#8494a8;background:rgba(132,148,168,.05)}
.empty{border:1px dashed #304157;border-radius:11px;padding:28px;text-align:center;color:#708197;font-size:10px}.evidence{border:1px solid #23354b;border-radius:11px;padding:12px;background:#0b1421;margin-top:8px}.evidencehead{display:flex;justify-content:space-between;gap:10px}.evidencetitle{font-size:10px;font-weight:800}.evidencemeta{font-size:8px;color:#6e829a}.snippet{font-size:10px;color:#9aabba;line-height:1.6;margin-top:7px}.confidence{display:flex;align-items:center;gap:8px;font-size:9px;color:#708299}.bar{height:4px;flex:1;background:#1a293b;border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.page{display:grid;grid-template-columns:1.1fr .75fr 1fr .65fr;gap:10px;padding:9px 0;border-bottom:1px solid #1a2839;align-items:center;font-size:9px}.page:last-child{border-bottom:0}.page small{display:block;color:#586a80;font-size:8px;margin-top:2px}.kv{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.kvitem{border:1px solid #1e2e42;border-radius:9px;padding:10px;background:#0b1421}.kvkey{font-size:8px;color:#60738b;text-transform:uppercase;letter-spacing:.07em}.kvvalue{font-size:10px;font-weight:750;margin-top:4px;overflow-wrap:anywhere}.timeline{border-left:1px solid #26384d;margin:4px 0 0 5px;padding-left:15px}.event{position:relative;padding:0 0 14px;font-size:9px;color:#8fa0b4}.event:before{content:"";position:absolute;left:-19px;top:3px;width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px #0e1623}.event b{color:#d9e2ed}.event small{display:block;color:#52647b;margin-top:3px}
.stTextInput input,.stTextArea textarea,.stNumberInput input,.stSelectbox div[data-baseweb=select]{background:#0b1421!important;color:#eaf1f8!important;border:1px solid #26374c!important;border-radius:9px!important}.stTextInput input:focus,.stTextArea textarea:focus{border-color:#4c7884!important;box-shadow:0 0 0 1px #4c7884!important}.stButton>button{border-radius:9px!important;min-height:38px!important;border:1px solid #293a50!important;background:#111d2c!important;color:#dce6f0!important;font-weight:750!important;font-size:10px!important}.stButton>button:hover{border-color:#4e687e!important;background:#162438!important}.stButton>button[kind=primary]{background:#4b83a0!important;border-color:#5a9bb7!important;color:#fff!important}.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--accent),var(--blue))!important}.stFileUploader{border:1px dashed #385069!important;border-radius:11px!important;background:#0b1421!important;padding:4px}.stFileUploader label{color:#91a2b6!important}.stExpander{border:1px solid #24364b!important;border-radius:10px!important;background:#0c1522!important}.stAlert{border-radius:9px!important}.caption{font-size:9px;color:#64758b}
@media(max-width:1100px){.grid4{grid-template-columns:repeat(2,1fr)}.twocol{grid-template-columns:1fr}.threecol{grid-template-columns:1fr}.microgrid{grid-template-columns:repeat(2,1fr)}}@media(max-width:700px){.block-container{padding:18px 14px 50px}.hero h1{font-size:26px}.grid4{grid-template-columns:1fr}.microgrid{grid-template-columns:1fr}.page{grid-template-columns:1fr 1fr}.topright{display:none}}
</style>''', unsafe_allow_html=True)


def sidebar(system):
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brandmark">BR</div><div><div class="brandname">BookRAG Medical</div><div class="brandsub">Evidence-first document research</div></div></div>', unsafe_allow_html=True)
        current = st.session_state.get("bookrag_page", "Home")
        for group, items in [("Workspace", ["Home", "Documents", "Live Processing", "Ask BookRAG"]), ("Research tools", ["Inspector", "System", "Settings"])]:
            st.markdown(f'<div class="navgroup">{group}</div>', unsafe_allow_html=True)
            for item in items:
                label = ("●  " if item == current else "○  ") + item
                if st.button(label, key=f"nav_{item}", use_container_width=True):
                    _navigate(item)
        st.markdown('<div class="sidefoot"><b>Local & private</b><br>Documents, retrieval and generation remain inside your configured local runtime.</div>', unsafe_allow_html=True)


def topbar(system, title: str):
    active = len(active_docs(system))
    st.markdown(f'<div class="top"><div><div class="crumb">BookRAG / Research workspace</div><div class="title">{_esc(title)}</div></div><div class="topright"><div class="chip"><span class="dot"></span>{active} processing</div><div class="chip">⌘K  Commands</div></div></div>', unsafe_allow_html=True)


def upload_block(system):
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Add research material</div><div class="sectionsub">Upload one or more PDFs. They are persisted and processing starts automatically.</div></div></div>', unsafe_allow_html=True)
    files = st.file_uploader("PDF documents", type=["pdf"], accept_multiple_files=True, key="premium_uploader", label_visibility="collapsed")
    if files:
        seen = st.session_state.setdefault("uploaded_hashes", set())
        added, errors = 0, []
        for f in files:
            payload = f.getvalue(); digest = hashlib.sha256(payload).hexdigest()
            if digest in seen: continue
            try:
                save_pdf(Path(system.settings.incoming_dir), f.name, payload); seen.add(digest); added += 1
            except Exception as exc: errors.append(f"{f.name}: {exc}")
        for e in errors: st.error(e)
        if added:
            jid = auto_ingest(system, added)
            if jid:
                st.session_state["bookrag_page"] = "Live Processing"
                st.success(f"{added} document(s) accepted · processing started")
                st.rerun()
            elif st.session_state.get("auto_ingest_error"):
                st.error(st.session_state["auto_ingest_error"])
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every="2s")
def global_live(system):
    active = active_docs(system); ready = ready_docs(system)
    label = f"{len(active)} document(s) actively processing" if active else (f"{len(ready)} document(s) ready" if ready else "Workspace idle")
    state = "PROCESSING" if active or _job_running() else ("READY" if ready else "IDLE")
    st.markdown(f'<div class="livebar"><span class="livepulse"></span><b>LIVE</b><span>{_esc(label)}</span><span class="grow"></span>{_status(state)}</div>', unsafe_allow_html=True)


def home(system):
    topbar(system, "Research workspace")
    st.markdown('<div class="hero"><div class="kicker">Medical document intelligence</div><h1>Research with evidence, not guesses.</h1><p>Turn medical PDFs into a private, searchable knowledge base. Follow extraction in real time, inspect every page, and ask questions with transparent evidence and abstention when the library cannot support an answer.</p><div class="hero-actions">', unsafe_allow_html=True)
    a, b = st.columns([1, 1])
    with a:
        if st.button("＋ Add PDFs", key="hero_add", type="primary", use_container_width=True): _navigate("Documents")
    with b:
        if st.button("Ask the library →", key="hero_ask", use_container_width=True): _navigate("Ask BookRAG")
    st.markdown('</div></div>', unsafe_allow_html=True)
    global_live(system)
    ds, rd, ac = docs(system), ready_docs(system), active_docs(system)
    st.markdown('<div class="grid4">', unsafe_allow_html=True)
    for label, value, hint in [("Documents",len(ds),"managed PDFs"),("Ready",len(rd),"grounded research sources"),("Processing",len(ac),"live pipeline jobs"),("Search sections",total_chunks(system),"indexed retrieval units")]:
        st.markdown(f'<div class="stat"><div class="statlabel">{label}</div><div class="statvalue">{value:,}</div><div class="stathint">{hint}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    twocol, side = st.columns([1.45,.75])
    with twocol:
        st.markdown('<div class="section"><div class="sectiontitle">Continue research</div><div class="sectionsub">Your most relevant workspace actions.</div>', unsafe_allow_html=True)
        for n, title, desc, target in [("01","Upload documents","Add PDFs and start automatic ingestion.","Documents"),("02","Watch processing","See exact page, stage and latest event.","Live Processing"),("03","Ask grounded questions","Retrieve evidence before generating an answer.","Ask BookRAG")]:
            st.markdown(f'<div class="evidence"><div class="evidencehead"><div class="evidencetitle">{n} · {title}</div><span class="evidencemeta">{target}</span></div><div class="snippet">{desc}</div></div>', unsafe_allow_html=True)
            if st.button(f"Open {title}", key=f"home_{n}"): _navigate(target)
        st.markdown('</div>', unsafe_allow_html=True)
    with side:
        st.markdown('<div class="section"><div class="sectiontitle">Trust model</div><div class="sectionsub">What BookRAG does before it answers.</div>', unsafe_allow_html=True)
        for t,d in [("Retrieve","Hybrid search finds relevant document evidence."),("Rerank","Evidence is ordered before generation."),("Ground","The answer is constrained by retrieved sources."),("Abstain","If evidence is insufficient, BookRAG can refuse to invent.")]:
            st.markdown(f'<div class="guide"><b>{t}</b><div class="muted">{d}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def documents_page(system):
    topbar(system, "Documents")
    upload_block(system)
    ds = docs(system)
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Document library</div><div class="sectionsub">Search and filter the persistent ingestion state.</div></div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,1,1])
    with c1: query=st.text_input("Search", placeholder="Search filename, hash or document ID", label_visibility="collapsed", key="doc_search")
    with c2: status=st.selectbox("Status", ["All","Ready","Processing","Failed"], label_visibility="collapsed", key="doc_status")
    with c3: sort=st.selectbox("Sort", ["Newest","Name","Status"], label_visibility="collapsed", key="doc_sort")
    rows=[]
    for d in ds:
        name=str(d.get("file_name") or d.get("filename") or d.get("document_id") or "Document")
        stt=str(d.get("status") or "UNKNOWN").upper()
        hay=(name+" "+str(d.get("document_id",""))).lower()
        if query and query.lower() not in hay: continue
        if status=="Ready" and stt not in {"READY","COMPLETED"}: continue
        if status=="Processing" and stt not in ACTIVE: continue
        if status=="Failed" and "FAILED" not in stt: continue
        rows.append(d)
    if sort=="Name": rows.sort(key=lambda x:str(x.get("file_name") or x.get("filename") or "").lower())
    elif sort=="Status": rows.sort(key=lambda x:str(x.get("status") or ""))
    else: rows.sort(key=lambda x:str(x.get("modified_at") or x.get("ingestion_started_at") or ""), reverse=True)
    if not rows: st.markdown('<div class="empty">No documents match this view.</div>', unsafe_allow_html=True)
    for d in rows:
        name=str(d.get("file_name") or d.get("filename") or d.get("document_id") or "Document"); p=_progress(d); stt=str(d.get("status") or "UNKNOWN")
        st.markdown(f'<div class="docrow"><div class="dochead"><div><div class="docname">{_esc(name)}</div><div class="meta">{_esc(d.get("document_id"))} · {_int(d.get("file_size")):,} bytes</div></div>{_status(stt)}</div><div class="meter"><i style="width:{p*100:.1f}%"></i></div><div class="microgrid"><div class="micro"><span>Stage</span><b>{_esc(_stage(d)[0])}</b></div><div class="micro"><span>Pages</span><b>{_int(d.get("current_page"))} / {_int(d.get("total_pages"))}</b></div><div class="micro"><span>Chunks</span><b>{_int(d.get("chunk_count")):,}</b></div><div class="micro"><span>Embeddings</span><b>{_int(d.get("embedding_count")):,}</b></div></div></div>', unsafe_allow_html=True)
        a,b=st.columns([1,5])
        with a:
            if st.button("Inspect", key=f"inspect_{d.get('document_id')}"): st.session_state["inspect_doc_id"]=d.get("document_id"); _navigate("Inspector")
    st.markdown('</div>', unsafe_allow_html=True)


@st.fragment(run_every="2s")
def processing_live(system):
    topbar(system, "Live Processing")
    active=active_docs(system)
    if not active:
        st.markdown('<div class="section"><div class="empty">No document is processing right now. Upload a PDF to start a live pipeline.</div></div>', unsafe_allow_html=True); return
    latest={str(e.get("document_id")):e for e in events(system,limit=600)}
    for d in active:
        did=str(d.get("document_id")); name=str(d.get("file_name") or d.get("filename") or did); stage,desc=_stage(d); p=_progress(d); current,total=_int(d.get("current_page")),_int(d.get("total_pages")); ev=latest.get(did,{})
        st.markdown(f'<div class="section"><div class="dochead"><div><div class="docname">{_esc(name)}</div><div class="meta">{_esc(stage)} · {_elapsed(d.get("ingestion_started_at")):.1f}s elapsed</div></div>{_status(d.get("status"))}</div><div class="sectionsub">{_esc(desc)}</div>', unsafe_allow_html=True)
        st.progress(p, text=f"Pipeline progress · {p*100:.0f}%")
        st.markdown(f'<div class="microgrid"><div class="micro"><span>Current page</span><b>{current} / {total or "—"}</b></div><div class="micro"><span>Stage</span><b>{_esc(stage)}</b></div><div class="micro"><span>Latest event</span><b>{_esc(ev.get("message") or ev.get("event_type") or "Working…")}</b></div><div class="micro"><span>Updated</span><b>{_esc(ev.get("created_at") or d.get("modified_at") or "now")}</b></div></div>', unsafe_allow_html=True)
        prs=pages(system,did)
        if prs:
            with st.expander(f"Page-level extraction · {len(prs)} records", expanded=False):
                done=sum(1 for x in prs if str(x.get("extraction_status","")).upper()=="COMPLETED")
                st.caption(f"{done} / {len(prs)} page records completed")
                for pg in prs:
                    st.markdown(f'<div class="page"><div><b>Page {_int(pg.get("page_number"))}</b><small>{_esc(pg.get("updated_at"))}</small></div><div>{_status(pg.get("extraction_status"))}</div><div>OCR · {_esc(pg.get("ocr_status") or "not required")}</div><div>{len(str(pg.get("text") or "")):,} chars</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def processing_page(system):
    processing_live(system)


def ask_page(system):
    topbar(system,"Ask BookRAG")
    available=ready_docs(system)
    st.markdown('<div class="section"><div class="sectiontitle">Grounded research</div><div class="sectionsub">Your answer is generated only after retrieval. If the library cannot support the question, the system can abstain.</div>',unsafe_allow_html=True)
    names=["All ready documents"]+[str(d.get("file_name") or d.get("filename") or "Document") for d in available]
    scope=st.selectbox("Source scope",names,key="ask_scope")
    selected=None if scope==names[0] else {"document_id":available[names.index(scope)-1].get("document_id")}
    q=st.text_area("Question",height=120,placeholder="What does the literature say about…?",key="research_question")
    a,b,c=st.columns([2,1,1])
    with a:
        run=st.button("Run grounded search",type="primary",use_container_width=True,key="ask_run")
    with b:
        if st.button("Clear",use_container_width=True): st.session_state.pop("answer_result",None); st.rerun()
    with c:
        if st.button("Inspector",use_container_width=True): _navigate("Inspector")
    if run:
        if not q.strip(): st.warning("Enter a research question first.")
        elif not available: st.warning("No ready document is available yet.")
        else:
            with st.spinner("Retrieving evidence and composing a grounded answer…"):
                try: st.session_state["answer_result"]=system.answer(q.strip(),metadata_filter=selected)
                except Exception as exc: st.error(f"The answer could not be produced safely: {exc}")
    ans=st.session_state.get("answer_result")
    if ans:
        st.markdown('</div>',unsafe_allow_html=True)
        answer=str(ans.get("answer") or "")
        st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Answer</div><div class="sectionsub">Evidence-grounded response · generated from the selected library.</div></div>{_status("ABSTAIN" if ans.get("abstained") else "GROUNDED")}</div><div style="font-size:13px;line-height:1.75;color:#dce6ef;white-space:pre-wrap">{_esc(answer)}</div></div>',unsafe_allow_html=True)
        evs=ans.get("evidence",[]) or ans.get("citations",[]) or []
        st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">Evidence</div><div class="sectionsub">Sources retrieved for this answer. Expand the query trace only when troubleshooting.</div></div><span class="chip">'+str(len(evs))+' sources</span></div>',unsafe_allow_html=True)
        for i,e in enumerate(evs,1):
            if isinstance(e,dict):
                title=e.get("file_name") or e.get("document_id") or f"Source {i}"; page=e.get("page_number") or e.get("page") or "—"; score=e.get("score") or e.get("rerank_score") or e.get("similarity") or "—"; snippet=e.get("text") or e.get("snippet") or e.get("content") or ""
            else: title=f"Source {i}"; page="—"; score="—"; snippet=str(e)
            st.markdown(f'<div class="evidence"><div class="evidencehead"><div class="evidencetitle">{_esc(title)}</div><div class="evidencemeta">Page {_esc(page)} · score {_esc(score)}</div></div><div class="snippet">{_esc(snippet)}</div></div>',unsafe_allow_html=True)
        with st.expander("Query trace & raw retrieval state"):
            st.json({"query_trace":ans.get("query_trace",{}),"citations":ans.get("citations",[]),"abstained":ans.get("abstained",False)})
        st.markdown('</div>',unsafe_allow_html=True)
    else: st.markdown('</div>',unsafe_allow_html=True)


def inspector_page(system):
    topbar(system,"Inspector")
    ds=docs(system)
    if not ds: st.markdown('<div class="section"><div class="empty">No document is available to inspect.</div></div>',unsafe_allow_html=True); return
    ids=[str(d.get("document_id")) for d in ds]; chosen=st.session_state.get("inspect_doc_id")
    default=ids.index(str(chosen)) if chosen in ids else 0
    idx=st.selectbox("Document",range(len(ds)),index=default,format_func=lambda i:str(ds[i].get("file_name") or ds[i].get("filename") or ids[i]),key="inspect_selector")
    d=ds[idx]; did=str(d.get("document_id")); stage,desc=_stage(d); p=_progress(d)
    st.markdown(f'<div class="section"><div class="sectionhead"><div><div class="sectiontitle">{_esc(d.get("file_name") or d.get("filename") or did)}</div><div class="sectionsub">{_esc(desc)}</div></div>{_status(d.get("status"))}</div><div class="meter"><i style="width:{p*100:.1f}%"></i></div>',unsafe_allow_html=True)
    vals=[("Stage",stage),("Pages",f"{_int(d.get('current_page'))} / {_int(d.get('total_pages'))}"),("Elapsed",f"{_elapsed(d.get('ingestion_started_at'),d.get('ingestion_completed_at')):.2f}s"),("Chunks",f"{_int(d.get('chunk_count')):,}"),("Embeddings",f"{_int(d.get('embedding_count')):,}"),("Embedding dimension",d.get("embedding_dimension") or "—"),("Parser",d.get("parser_version") or "—"),("Version",d.get("version_id") or d.get("content_hash") or "—")]
    st.markdown('<div class="kv">'+''.join(f'<div class="kvitem"><div class="kvkey">{_esc(k)}</div><div class="kvvalue">{_esc(v)}</div></div>' for k,v in vals)+'</div>',unsafe_allow_html=True)
    if d.get("error"): st.error(str(d.get("error")))
    if st.button("Verify index",type="primary",use_container_width=True,key="verify_index"):
        try: st.session_state["verify_result"]=system.verify_index(did)
        except Exception as exc: st.error(str(exc))
    if st.session_state.get("verify_result") is not None: st.json(st.session_state.pop("verify_result"))
    st.markdown('</div>',unsafe_allow_html=True)
    prs=pages(system,did)
    st.markdown('<div class="section"><div class="sectiontitle">Page records</div><div class="sectionsub">Extraction and OCR state persisted by the ingestion pipeline.</div>',unsafe_allow_html=True)
    if prs:
        for pg in prs: st.markdown(f'<div class="page"><div><b>Page {_int(pg.get("page_number"))}</b><small>{_esc(pg.get("updated_at"))}</small></div><div>{_status(pg.get("extraction_status"))}</div><div>OCR · {_esc(pg.get("ocr_status") or "not required")}</div><div>{len(str(pg.get("text") or "")):,} chars</div></div>',unsafe_allow_html=True)
    else: st.markdown('<div class="empty">No page records stored yet.</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
    with st.expander("Event timeline"):
        es=events(system,did,250); st.markdown('<div class="timeline">',unsafe_allow_html=True)
        for e in reversed(es[-40:]): st.markdown(f'<div class="event"><b>{_esc(e.get("message") or e.get("event_type") or e.get("stage") or "Event")}</b> · {_status(e.get("status") or e.get("stage"))}<small>{_esc(e.get("created_at"))} · page {_esc(e.get("current_page") or "—")} / {_esc(e.get("total_pages") or "—")}</small></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with st.expander("Raw state (advanced)"): st.json(d)


@st.fragment(run_every="8s")
def system_snapshot(system):
    try: report=system.health_report()
    except Exception as exc: report={"ready":False,"error":str(exc)}
    try: ok,msg,models=ollama_health(system.settings.ollama_base_url)
    except Exception as exc: ok,msg,models=False,str(exc),[]
    emb=report.get("embedding",{}); idx=report.get("index",{}); contract=report.get("feature_contract",report.get("pipeline",{}))
    rows=[("Ollama","ONLINE" if ok else "OFFLINE",msg),("Embedding","PASS" if emb.get("ok") else "UNKNOWN",emb.get("identity") or emb.get("error") or system.settings.embedding_model),("Vector index",str(idx.get("status","UNKNOWN")).upper(),idx.get("error") or "Runtime-reported index health"),("Production contract","PASS" if contract.get("all_resolved",False) else "CHECK","Feature resolution")]
    st.markdown('<div class="section"><div class="sectionhead"><div><div class="sectiontitle">System health</div><div class="sectionsub">Live operational signals · refreshed automatically.</div></div><span class="chip"><span class="dot"></span>8s</span></div>',unsafe_allow_html=True)
    for label,status,detail in rows: st.markdown(f'<div class="page"><div><b>{_esc(label)}</b></div><div>{_status(status)}</div><div>{_esc(detail)}</div><div></div></div>',unsafe_allow_html=True)
    if models: st.caption("Ollama models · "+", ".join(models))
    st.markdown('</div>',unsafe_allow_html=True)


def system_page(system):
    topbar(system,"System")
    system_snapshot(system)
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="section"><div class="sectiontitle">Runtime profile</div><div class="sectionsub">Configured local services.</div>',unsafe_allow_html=True)
        for k,v in [("Ollama",system.settings.ollama_base_url),("Embedding",system.settings.embedding_model),("Generation",system.settings.generation_model),("Top K",system.settings.top_k)]: st.markdown(f'<div class="kvitem" style="margin-top:7px"><div class="kvkey">{k}</div><div class="kvvalue">{_esc(v)}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="section"><div class="sectiontitle">Operational guidance</div><div class="sectionsub">How to interpret this workspace.</div>',unsafe_allow_html=True)
        for k,v in [("Ready","Document is published to retrieval."),("Processing","Persistent page state is still changing."),("Failed","Inspect the document error and event timeline."),("Abstain","The evidence did not justify a confident answer.")]: st.markdown(f'<div class="evidence"><div class="evidencetitle">{k}</div><div class="snippet">{v}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)


def settings_page(system):
    topbar(system,"Settings")
    s=system.settings
    st.markdown('<div class="section"><div class="sectiontitle">Research behavior</div><div class="sectionsub">Safe controls are exposed here; security validation remains in the application layer.</div>',unsafe_allow_html=True)
    with st.form("settings_form"):
        host=st.text_input("Ollama host",str(s.ollama_base_url)); emb=st.text_input("Embedding model",str(s.embedding_model)); gen=st.text_input("Generation model",str(s.generation_model))
        a,b=st.columns(2)
        with a: topk=st.number_input("Retrieval results",1,50,int(s.top_k)); vw=st.slider("Semantic weight",0.,1.,float(s.vector_weight),.05)
        with b: temp=st.slider("Generation temperature",0.,1.,float(s.temperature),.05); neighbor=st.checkbox("Neighbor expansion",bool(s.neighbor_expansion))
        save=st.form_submit_button("Save configuration",type="primary",use_container_width=True)
    if save:
        try:
            ok,w=system.apply_settings_in_place({"ollama_base_url":host,"embedding_model":emb,"generation_model":gen,"top_k":int(topk),"vector_weight":float(vw),"temperature":float(temp),"neighbor_expansion":bool(neighbor)})
            if ok: st.success("Configuration saved.")
            for x in w or []: st.warning(x)
        except Exception as exc: st.error(f"Settings were not saved: {exc}")
    st.markdown('</div>',unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="BookRAG Medical",page_icon="BR",layout="wide",initial_sidebar_state="expanded")
    st.session_state.setdefault("bookrag_page","Home"); st.session_state.setdefault("uploaded_hashes",set())
    system=get_system(); css(); sidebar(system)
    page=st.session_state.get("bookrag_page","Home")
    {"Home":home,"Documents":documents_page,"Live Processing":processing_page,"Ask BookRAG":ask_page,"Inspector":inspector_page,"System":system_page,"Settings":settings_page}.get(page,home)(system)


if __name__ == "__main__": main()
