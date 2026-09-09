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

PAGES = ["Overview", "Documents", "Index them", "Ingestion", "Inspector", "Settings", "Chat", "Health", "Background"]
ACTIVE_STAGES = {"RUNNING", "DISCOVERED", "VALIDATING", "EXTRACTING", "OCR", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING_INDEX", "BUILDING"}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
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
    return [item for item in docs(system) if str(item.get("status", "")).upper() == "READY"]


def active_document_count(system) -> int:
    return sum(str(item.get("status", "")).upper() in ACTIVE_STAGES for item in docs(system))


def start_ingestion(system, source_dir: str) -> str:
    folder = Path(source_dir).expanduser()
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    if not any(path.is_file() and path.suffix.lower() == ".pdf" for path in folder.iterdir()):
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
        payload = response.json()
        models = [str(item.get("name")) for item in payload.get("models", []) if item.get("name")] if isinstance(payload, dict) else []
        return True, "Ollama is reachable.", models
    except Exception as exc:
        return False, str(exc), []


def _status_class(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    if text in {"READY", "PASS", "COMPLETED", "HEALTHY", "OK"}:
        return "good"
    if text in {"FAILED", "FAIL", "ERROR", "UNAVAILABLE", "ABSTAIN"}:
        return "bad"
    if text in {"RUNNING", "PROCESSING", "BUILDING", "WARN", "WARNING"}:
        return "warn"
    return "neutral"


def _status(value: Any) -> str:
    text = str(value or "UNKNOWN")
    return f'<span class="status { _status_class(text) }">{_esc(text)}</span>'


def _go(page: str) -> None:
    st.session_state["studio_nav"] = page if page in PAGES else "Overview"
    st.rerun()


def css() -> None:
    st.markdown('''<style>
:root{--bg:#080d18;--surface:#111a2a;--stroke:#273449;--text:#f7f9fc;--muted:#8794a8}
html,body{overflow-x:hidden!important}.stApp{background:radial-gradient(900px 500px at 76% -8%,rgba(124,116,255,.16),transparent 58%),radial-gradient(650px 400px at 8% 10%,rgba(66,214,208,.08),transparent 62%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{display:none}[data-testid="stAppViewContainer"]{overflow:visible!important}[data-testid="stAppViewContainer"]>section.main{min-width:0!important}
.block-container{box-sizing:border-box!important;width:100%!important;max-width:1440px!important;margin:0 auto!important;padding:32px 32px 64px!important;min-width:0!important}
[data-testid="stSidebar"]{z-index:100!important}[data-testid="stSidebar"]>div:first-child{box-sizing:border-box!important;height:100vh!important;max-height:100vh!important;overflow-y:auto!important;overflow-x:hidden!important;padding:20px 18px!important}
.brand{display:flex;align-items:center;gap:12px;margin:4px 8px 24px}.brand-mark{width:44px;height:44px;border-radius:13px;display:grid;place-items:center;background:linear-gradient(135deg,#8177ff,#45d8d0);font-size:21px}.brand-name{font-weight:850;font-size:16px}.brand-sub{font-size:10px;color:#718097;margin-top:2px}.nav-label{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:#58677e;font-weight:850;margin:18px 8px 7px}.sidebar-divider{height:1px;background:var(--stroke);margin:16px 0}
[data-testid="stSidebar"] .stButton>button{width:100%!important;height:42px!important;min-height:42px!important;border:1px solid transparent!important;background:transparent!important;border-radius:10px!important;text-align:left!important;padding:0 12px!important;color:#98a5b8!important;font-size:13px!important;font-weight:650!important}[data-testid="stSidebar"] .stButton>button:hover{background:#151f31!important;color:#fff!important;border-color:#29364b!important}
.topbar{margin-bottom:24px}.eyebrow{font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:#6b7890}.page-title{font-size:30px;line-height:1.08;font-weight:850;letter-spacing:-.04em;margin-top:6px}.page-subtitle{font-size:12px;color:#9ca9bd;margin-top:8px}
.card{border:1px solid var(--stroke);border-radius:16px;background:rgba(17,26,42,.82);padding:20px;box-shadow:0 12px 32px rgba(0,0,0,.12);min-width:0}.card-title{font-size:14px;font-weight:820}.muted{color:var(--muted);font-size:11px}.metric-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:16px}.metric{border:1px solid #26344a;border-radius:13px;background:rgba(8,13,24,.26);padding:15px;text-align:center;min-width:0}.metric-value{font-size:38px;line-height:1;font-weight:850}.metric-label{font-size:11px;color:#8794a8;margin-top:8px}.data-wrap{max-width:100%;overflow:auto;border:1px solid #26344a;border-radius:11px;margin-top:14px}.data-table{width:100%;min-width:760px;border-collapse:collapse;font-size:10px}.data-table th,.data-table td{padding:9px 10px;border-bottom:1px solid #202d40;text-align:left;white-space:nowrap}.data-table th{color:#65748b;font-size:9px;text-transform:uppercase;letter-spacing:.06em;background:#101927}.data-table td{color:#cbd4e1}.event{display:grid;grid-template-columns:1fr auto;gap:10px;padding:9px 0;border-bottom:1px solid #202d40}.event:last-child{border-bottom:0}.event-copy{font-size:10px}.event-meta{font-size:9px;color:#758399}.status{display:inline-flex;align-items:center;border-radius:999px;padding:4px 8px;border:1px solid currentColor;font-size:9px;font-weight:850}.status.good{color:#63e5a3;background:rgba(57,217,138,.07)}.status.warn{color:#f5c867;background:rgba(244,189,85,.07)}.status.bad{color:#ff8492;background:rgba(255,104,123,.07)}.status.neutral{color:#a1aec0;background:rgba(135,148,168,.07)}
.answer,.glass-note{max-width:100%;overflow-wrap:anywhere;word-break:break-word}.answer{border:1px solid rgba(124,116,255,.3);background:rgba(124,116,255,.055);border-radius:12px;padding:13px;margin-top:14px}.answer-text{font-size:12px;line-height:1.6}.glass-note{border:1px dashed #35445b;border-radius:11px;padding:11px;color:#8795a9;font-size:11px;line-height:1.5;margin-top:12px}.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"],.stNumberInput input{background:#0e1725!important;color:#f4f7fb!important;border-color:#2b3950!important}.stFileUploader{background:#0e1725!important;border:1px dashed #3b4a62!important;border-radius:11px!important}
@media(max-width:1100px){.block-container{padding:28px 22px 56px!important}.metric-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:760px){.block-container{padding:20px 14px 44px!important}.page-title{font-size:25px}.metric-grid{grid-template-columns:1fr}.data-table{min-width:650px}}
</style>''', unsafe_allow_html=True)


def sidebar(system) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">📚</div><div><div class="brand-name">BookRAG Studio</div><div class="brand-sub">Local document intelligence</div></div></div>', unsafe_allow_html=True)
        current = st.session_state.get("studio_nav", "Overview")
        st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        for item in ["Overview", "Documents", "Index them", "Ingestion", "Inspector", "Settings"]:
            if st.button(("●  " if current == item else "○  ") + item, key=f"nav_{item}", use_container_width=True):
                _go(item)
        st.markdown('<div class="nav-label">Tools</div>', unsafe_allow_html=True)
        for item in ["Chat", "Health", "Background"]:
            if st.button(("●  " if current == item else "○  ") + item, key=f"nav_secondary_{item}", use_container_width=True):
                _go(item)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Documents</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="exact_uploads")
        if uploads:
            saved = st.session_state.setdefault("saved_pdf_hashes", set())
            incoming = Path(system.settings.incoming_dir)
            added = 0
            for item in uploads:
                payload = item.getvalue(); digest = hashlib.sha256(payload).hexdigest()
                if digest in saved:
                    continue
                try:
                    save_pdf(incoming, item.name, payload); saved.add(digest); added += 1
                except Exception as exc:
                    st.error(str(exc))
            if added:
                st.success(f"Added {added} PDF{'s' if added != 1 else ''} to the incoming folder.")
        st.text_input("Incoming folder", value=str(system.settings.incoming_dir), key="exact_incoming")
        if st.button("Start indexing", key="exact_start", use_container_width=True):
            try:
                st.session_state["studio_last_job"] = start_ingestion(system, st.session_state["exact_incoming"])
                _go("Ingestion")
            except Exception as exc:
                st.error(str(exc))
        if st.button("Recreate runtime", key="exact_recreate", use_container_width=True):
            get_system.clear(); _go("Overview")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        phrase = st.text_input("Type CLEAR ALL PDF DATA to enable deletion", key="bookrag_clear_phrase", type="password", placeholder="CLEAR ALL PDF DATA")
        if st.button("Clear all PDF data", key="exact_clear", use_container_width=True, disabled=phrase.strip() != "CLEAR ALL PDF DATA"):
            try:
                system.clear_pdf_data()
                st.session_state["bookrag_clear_phrase"] = ""
                st.session_state["saved_pdf_hashes"] = set()
                st.session_state.pop("exact_answer", None); st.session_state.pop("console_answer", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")


def topbar() -> None:
    st.markdown('<div class="topbar"><div class="eyebrow">BookRAG Studio</div><div class="page-title">Document Intelligence Workspace</div><div class="page-subtitle">Upload, index, search, inspect, and verify your local PDF knowledge base.</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💬 Chat", key="top_chat", use_container_width=True): _go("Chat")
    with c2:
        if st.button("🩺 Health", key="top_health", use_container_width=True): _go("Health")
    with c3:
        if st.button("↻ Refresh", key="top_refresh", use_container_width=True): _refresh()


def _refresh() -> None:
    st.rerun()


def overview(system) -> None:
    items = docs(system); ready = ready_docs(system); active = active_document_count(system)
    vectors = sum(_safe_int(item.get("vector_chunks", item.get("chunks", 0))) for item in items)
    st.markdown(f'<div class="card"><div class="eyebrow">Overview</div><div class="card-title">Your local RAG workspace</div><div class="muted">All primary UI actions use the same cached production system and shared navigation state.</div><div class="metric-grid"><div class="metric"><div class="metric-value">{len(items)}</div><div class="metric-label">Documents</div></div><div class="metric"><div class="metric-value">{len(ready)}</div><div class="metric-label">Ready</div></div><div class="metric"><div class="metric-value">{active}</div><div class="metric-label">Processing</div></div></div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        st.markdown('<div class="card"><div class="card-title">Ask Your Documents</div><div class="muted">Search and answer through the production RAG pipeline.</div>', unsafe_allow_html=True)
        question = st.text_area("Question", placeholder="Ask something about your indexed PDFs…", height=110, key="exact_overview_q", label_visibility="collapsed")
        if st.button("Search and answer", key="exact_overview_ask", use_container_width=True):
            if not question.strip(): st.warning("Please enter a question.")
            elif not ready: st.warning("No READY documents are available yet. Upload and index a PDF first.")
            else:
                try: st.session_state["exact_answer"] = system.answer(question.strip())
                except Exception as exc: st.error(f"Answer failed safely: {exc}")
        result = st.session_state.get("exact_answer")
        if result:
            st.markdown(f'<div class="answer"><div class="muted">Answer</div><div class="answer-text">{_esc(result.get("answer", ""))}</div></div>', unsafe_allow_html=True)
            for citation in result.get("citations", []) or []: st.write(citation)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="card-title">Runtime summary</div>', unsafe_allow_html=True)
        s = system.settings
        for label, value in [("Generation", s.generation_model), ("Embedding", s.embedding_model), ("Vector chunks", vectors), ("Top-k", s.top_k)]:
            st.markdown(f'<div class="event"><div class="event-copy">{_esc(label)}</div><div>{_esc(value)}</div></div>', unsafe_allow_html=True)
        if st.button("Open documents", key="overview_documents", use_container_width=True): _go("Documents")
        if st.button("Open ingestion", key="overview_ingestion", use_container_width=True): _go("Ingestion")
        st.markdown('</div>', unsafe_allow_html=True)


def documents(system) -> None:
    items = docs(system)
    st.markdown('<div class="card"><div class="card-title">Documents</div><div class="muted">Shared document state and ingestion metadata.</div>', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="glass-note">No documents have been uploaded yet.</div>', unsafe_allow_html=True)
    else:
        headers = ["Status", "File", "Pages", "Chunks", "Embeddings", "Dimension", "Stage", "Error"]
        st.markdown('<div class="data-wrap"><table class="data-table"><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>', unsafe_allow_html=True)
        for item in items:
            row = [item.get("status", "N/A"), item.get("file_name", item.get("filename", "N/A")), item.get("total_pages", item.get("pages", "N/A")), item.get("chunk_count", item.get("chunks", item.get("vector_chunks", "N/A"))), item.get("embedding_count", item.get("embeddings", "N/A")), item.get("embedding_dimension", item.get("dimension", "N/A")), item.get("current_stage", item.get("stage", "N/A")), item.get("error", "")]
            st.markdown('<tr>' + ''.join(f'<td>{_status(value) if i == 0 else _esc(value)}</td>' for i, value in enumerate(row)) + '</tr>', unsafe_allow_html=True)
        st.markdown('</table></div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        if st.button("Open index", key="documents_index", use_container_width=True): _go("Index them")
    with b:
        if st.button("Refresh documents", key="exact_documents_refresh", use_container_width=True): _refresh()
    st.markdown('</div>', unsafe_allow_html=True)


def chat(system) -> None:
    available = ready_docs(system)
    st.markdown('<div class="card"><div class="card-title">Chat</div><div class="muted">Ask the shared production RAG system and inspect citations/evidence.</div>', unsafe_allow_html=True)
    names = ["All ready documents"] + [str(item.get("file_name", item.get("filename", "Document"))) for item in available]
    scope = st.selectbox("Search scope", names, key="exact_scope")
    selected_filter = None if scope == names[0] else available[names.index(scope) - 1].get("document_id")
    nonce = st.session_state.get("studio_chat_nonce", 0)
    question = st.text_area("Question", placeholder="Ask a question and inspect the evidence behind the answer…", height=110, key=f"studio_question_{nonce}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Search and answer", key="exact_chat_search", use_container_width=True):
            if not question.strip(): st.warning("Please enter a question.")
            elif not available: st.warning("No READY documents are available yet.")
            else:
                try: st.session_state["console_answer"] = system.answer(question.strip(), metadata_filter={"document_id": selected_filter} if selected_filter else None)
                except Exception as exc: st.error(f"Answer failed safely: {exc}")
    with c2:
        if st.button("Clear chat", key="exact_chat_clear", use_container_width=True):
            st.session_state.pop("console_answer", None); st.session_state["studio_chat_nonce"] = nonce + 1; st.rerun()
    result = st.session_state.get("console_answer")
    if result:
        st.markdown(f'<div class="answer"><div class="muted">Answer</div><div class="answer-text">{_esc(result.get("answer", ""))}</div></div>', unsafe_allow_html=True)
        for citation in result.get("citations", []) or []: st.write(citation)
        with st.expander("Evidence and trace"): st.json({"evidence": result.get("evidence", []), "query_trace": result.get("query_trace", {})})
    st.markdown('</div>', unsafe_allow_html=True)


def health(system) -> None:
    try: report = system.health_report()
    except Exception as exc: report = {"ready": False, "embedding": {"ok": False, "error": str(exc)}, "index": {"status": "UNAVAILABLE", "error": str(exc)}, "audit": {"ok": False, "error": str(exc)}, "feature_contract": {"all_resolved": False}}
    ollama_ok, ollama_message, models = ollama_health(system.settings.ollama_base_url)
    embedding = report.get("embedding", {}); index = report.get("index", {}); audit = report.get("audit", {}); contract = report.get("feature_contract", {})
    index_status = str(index.get("status", "UNAVAILABLE")).upper(); embedding_ok = bool(embedding.get("ok")); audit_ok = bool(audit.get("ok", audit.get("status") in {"PASS", "READY", "OK"})); contract_ok = bool(contract.get("all_resolved", False))
    rows = [("Ollama", "PASS" if ollama_ok else "FAIL", ollama_message), ("Embedding", "PASS" if embedding_ok else "FAIL", embedding.get("identity") or embedding.get("error") or system.settings.embedding_model), ("Vector index", "READY" if index_status in {"READY", "OK"} else index_status, index.get("error") or index_status), ("Production contract", "PASS" if contract_ok else "FAIL", "All production features resolved" if contract_ok else "Production feature contract incomplete"), ("Index audit", "PASS" if audit_ok else "FAIL", audit.get("error") or ("Consistency checks complete" if audit_ok else "Audit failed"))]
    overall = bool(report.get("ready")) and ollama_ok
    st.markdown(f'<div class="card"><div class="card-title">System Health</div><div class="muted">Overall readiness: {_esc("READY" if overall else "ATTENTION REQUIRED")}</div><div class="data-wrap"><table class="data-table"><tr><th>Service</th><th>Status</th><th>Detail</th></tr>', unsafe_allow_html=True)
    for label, status, detail in rows: st.markdown(f'<tr><td>{_esc(label)}</td><td>{_status(status)}</td><td>{_esc(detail)}</td></tr>', unsafe_allow_html=True)
    st.markdown('</table></div>', unsafe_allow_html=True)
    if models: st.write(models)
    if st.button("Recheck health", key="exact_health_recheck", use_container_width=True): _refresh()
    st.markdown('</div>', unsafe_allow_html=True)


def ingestion(system, *, index_mode: bool = False) -> None:
    registry = get_jobs()
    with registry["lock"]: jobs = list(registry["items"].values())
    running = next((job for job in reversed(jobs) if job.get("status") == "RUNNING"), None)
    title = "Index them" if index_mode else "Ingestion"
    st.markdown(f'<div class="card"><div class="card-title">{title}</div><div class="muted">This job state is shared with Background and Documents.</div>', unsafe_allow_html=True)
    if running:
        elapsed = max(0.0, time.time() - float(running.get("started", time.time())))
        st.info(f"Active job · {Path(running.get('source_dir', '')).name} · {elapsed:.1f}s")
    else:
        st.caption("No ingestion worker is currently running.")
    if index_mode and st.button("Start indexing current incoming folder", key="index_mode_start", use_container_width=True):
        try: st.session_state["studio_last_job"] = start_ingestion(system, str(system.settings.incoming_dir)); _go("Ingestion")
        except Exception as exc: st.error(str(exc))
    rows = docs(system)
    if rows:
        st.markdown('<div class="data-wrap"><table class="data-table"><tr><th>Status</th><th>File</th><th>Stage</th><th>Page</th><th>Chunks</th><th>Embeddings</th><th>Dimension</th><th>Error</th></tr>', unsafe_allow_html=True)
        for item in rows:
            values = [item.get("status", "N/A"), item.get("file_name", item.get("filename", "N/A")), item.get("current_stage", item.get("stage", "N/A")), item.get("current_page", item.get("page", "N/A")), item.get("chunks", item.get("vector_chunks", "N/A")), item.get("embedding_count", item.get("embeddings", "N/A")), item.get("embedding_dimension", item.get("dimension", "N/A")), item.get("error", "")]
            st.markdown('<tr>' + ''.join(f'<td>{_status(value) if idx == 0 else _esc(value)}</td>' for idx, value in enumerate(values)) + '</tr>', unsafe_allow_html=True)
        st.markdown('</table></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="glass-note">No document records are available yet.</div>', unsafe_allow_html=True)
    last = next((job for job in reversed(jobs) if job.get("status") in {"COMPLETED", "FAILED"}), None)
    if last and last.get("status") == "COMPLETED": st.success("Ingestion completed successfully.")
    elif last and last.get("status") == "FAILED": st.error(str(last.get("error") or "Ingestion failed."))
    a, b = st.columns(2)
    with a:
        if st.button("Refresh progress", key=f"exact_ingestion_refresh_{'index' if index_mode else 'main'}", use_container_width=True): _refresh()
    with b:
        if st.button("Open documents", key=f"exact_ingestion_docs_{'index' if index_mode else 'main'}", use_container_width=True): _go("Documents")
    st.markdown('</div>', unsafe_allow_html=True)


def background(system) -> None:
    registry = get_jobs()
    with registry["lock"]: jobs = list(registry["items"].values())
    st.markdown('<div class="card"><div class="card-title">Background workers</div><div class="muted">Shared ingestion worker registry.</div>', unsafe_allow_html=True)
    if not jobs: st.markdown('<div class="glass-note">No background ingestion jobs have been created yet.</div>', unsafe_allow_html=True)
    for job in reversed(jobs): st.markdown(f'<div class="event"><div class="event-copy">{_esc(Path(job.get("source_dir", "")).name)}<div class="event-meta">{_esc(job.get("error") or "ingestion")}</div></div>{_status(job.get("status"))}</div>', unsafe_allow_html=True)
    if st.button("Refresh workers", key="exact_background_refresh", use_container_width=True): _refresh()
    st.markdown('</div>', unsafe_allow_html=True)


def inspector(system) -> None:
    items = docs(system)
    st.markdown('<div class="card"><div class="card-title">Document and index inspector</div>', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="glass-note">No documents are indexed yet.</div>', unsafe_allow_html=True)
    else:
        labels = [str(item.get("file_name", item.get("filename", item.get("document_id", "Document")))) for item in items]
        selected = items[st.selectbox("Document", range(len(items)), format_func=lambda i: labels[i], key="exact_inspector_doc")]
        for label, value in [("Status", selected.get("status")), ("Pages", selected.get("total_pages", selected.get("pages"))), ("Dimension", selected.get("embedding_dimension", selected.get("dimension"))), ("Version", selected.get("version_id", selected.get("version"))), ("Chunks", selected.get("chunk_count", selected.get("chunks"))), ("Document ID", selected.get("document_id"))]: st.markdown(f'<div class="event"><div class="event-copy">{_esc(label)}</div><div>{_esc(value)}</div></div>', unsafe_allow_html=True)
        if st.button("Verify index", key="exact_verify", use_container_width=True):
            try: st.json(system.verify_index(selected.get("document_id")))
            except Exception as exc: st.error(f"Index verification failed: {exc}")
        with st.expander("Page checkpoints"): st.json(selected.get("page_checkpoints", []))
        with st.expander("Raw metadata"): st.json(selected)
    st.markdown('</div>', unsafe_allow_html=True)


def settings(system) -> None:
    current = system.settings
    st.markdown('<div class="card"><div class="card-title">Settings</div><div class="muted">Validated live configuration shared by the whole application.</div>', unsafe_allow_html=True)
    with st.form("exact_settings_form"):
        host = st.text_input("Ollama host", value=str(current.ollama_base_url)); embedding = st.text_input("Embedding model", value=str(current.embedding_model)); generation = st.text_input("Generation model", value=str(current.generation_model))
        c1, c2 = st.columns(2)
        with c1: chunk = st.number_input("Chunk size", 200, 4000, int(current.chunk_size), 50); top_k = st.number_input("Top-k", 1, 50, int(current.top_k), 1); vector_weight = st.slider("Vector weight", 0.0, 1.0, float(current.vector_weight), 0.05)
        with c2: overlap = st.number_input("Chunk overlap", 0, 3999, int(current.chunk_overlap), 10); temperature = st.slider("Temperature", 0.0, 1.0, float(current.temperature), 0.05); neighbor = st.checkbox("Neighbor expansion", value=bool(current.neighbor_expansion))
        apply = st.form_submit_button("Apply live settings", use_container_width=True)
    if apply:
        if overlap >= chunk: st.error("Chunk overlap must be smaller than chunk size.")
        else:
            updates = {"ollama_base_url": host, "embedding_model": embedding, "generation_model": generation, "chunk_size": int(chunk), "chunk_overlap": int(overlap), "top_k": int(top_k), "temperature": float(temperature), "vector_weight": float(vector_weight), "neighbor_expansion": bool(neighbor)}
            try: st.success(f"Settings applied: {system.apply_settings_in_place(updates)}")
            except Exception as exc: st.error(f"Settings could not be applied: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="BookRAG Studio", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
    st.session_state.setdefault("studio_nav", "Overview"); st.session_state.setdefault("studio_chat_nonce", 0)
    system = get_system(); css(); sidebar(system); topbar()
    page = st.session_state.get("studio_nav", "Overview")
    if page == "Overview": overview(system)
    elif page == "Documents": documents(system)
    elif page == "Index them": ingestion(system, index_mode=True)
    elif page == "Ingestion": ingestion(system)
    elif page == "Inspector": inspector(system)
    elif page == "Settings": settings(system)
    elif page == "Chat": chat(system)
    elif page == "Health": health(system)
    elif page == "Background": background(system)
    else: st.session_state["studio_nav"] = "Overview"; st.rerun()


if __name__ == "__main__": main()
